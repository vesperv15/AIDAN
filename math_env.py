import torch
from sympy import symbols, simplify, expand, factor, diff, integrate, sin, cos, exp, log, Integer, count_ops
from sympy.parsing.mathematica import parse_mathematica
import random
import numpy as np
from lean_validator import LeanValidator
from sympy import Symbol, srepr

x, y, z, t = symbols('x y z t')

ACTION_TOKENS = [
    'SIMPLIFY', 'EXPAND', 'FACTOR', 'DIFFERENTIATE_X', 'TERMINATE', 'UNDO'
]

BASE_TOKENS = [
    'x', 'y', 'z', 't', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '**', '(', ')', 'sin', 'cos', 'exp', 'log', 'tan',
    '=', 'WRT', '^', '<UNK>', '<PAD>'
]

class MathEnv:
    def __init__(self, max_steps=20):
        # düşünmee için daha fazla zaman 15>20
        self.x = x
        self.y = y
        self.max_steps = max_steps
        self.current_step = 0
        self.current_expression = None
        self.target_expression = None
        self.last_error = None
        self._VOCAB = {token: i for i, token in enumerate(BASE_TOKENS + ACTION_TOKENS)}
        self.MAX_SEQ_LEN = 128
        self.problem_string = None
        self.validator= LeanValidator()
        self.history= []
        self.visited_states= set()

    @property
    def action_space_size(self):
        return len(ACTION_TOKENS)

    @property
    def vocab_size(self):
        return len(self._VOCAB)
    
    def get_context_type(self):
        """
        Mevcut problemin matematiksel bağlamını (kategorisini) döndürür.
        LeanValidator'ın doğru mathlib kütüphanelerini yüklemesini sağlar.
        """
        #bağlam atanmadıysa varsayılan basic algebra
        if hasattr(self, 'current_context'):
            return self.current_context
        return "basic_algebra"
    
    def _generate_algebra_problem(self):
        """
        Zorluk Seviyesi 1 (Cebir) için problemler üretir.
        Modelden 'EXPAND' (genişletme) veya 'FACTOR' (çarpanlara ayırma)
        gibi hamleler yapmasını beklediğimiz ifadeler oluşturur.
        """
        # katsayılar rastgele a v e c 0 olması engellendi
        a = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        b = random.randint(-5, 5)
        c = random.choice([-2, -1, 1, 2])
        d = random.randint(-5, 5)
        
        if random.random() > 0.5:
            #1. secenek: (ax + b) * (cx + d) beklenen expand
            expr = (a * self.x + b) * (c * self.x + d)
        else:
            #2.secenek: sadeleştirme ya d atürev içib bir polinom
            expr = a * self.x**2 + b * self.x + c * d
            
        return expr
        
    def _generate_basic_problem(self):
        """aıdan için basit problem üreitimi"""
        a = random.randint(1, 10)
        b= random.randint(1, 20)
        c= random.randint(20, 50)
        
        self.target_expression= Integer(0)
        problem_expr= a*self.x + b -c
        
        return problem_expr
    def _generate_complex_problem(self, complexity):
        """
        Muhakemeyi zorlaştıran rastgele soru üreticisi.
        Artık sadece x değil, x ve sayıların kombinasyonları üretiliyor.
        """
        ops = ['+', '-', '*']
        num_terms = random.randint(2, 3)
        
        expr = Integer(random.randint(1, 20))
        
        for _ in range(num_terms - 1):
            op = random.choice(ops)
            next_val = random.randint(1, 10)
            
            if op == '+':
                expr += next_val * self.x
            elif op == '-':
                expr -= next_val
            elif op == '*':
                expr *= random.choice([self.x, next_val])

        self.target_expression = simplify(expr)
        
        start_expr = expand(expr + random.randint(1, 5))
        return str(start_expr)

    def get_skeleton(self,expr):
        """Denklemi değişkenlerden bağımsız bir iskelete çevirir. x**2 => var**2"""
        try:
            #tüm değişkenleri bulalım ve sıralayım.
            symbols= sorted(list(expr.free_symbols), key=lambda s: s.name)
            skeleton_expr= expr
            #değişkenleri var şeklinde standartlaştır
            
            for i, sym in enumerate(symbols):
                skeleton_expr= skeleton_expr.subs(sym, Symbol(f'VAR{i}'))
            #srepr en güvenli iskelet
            return srepr(skeleton_expr)
        except Exception:
            return str(expr)#hata durumunda ham hali döndür
    
    def _tokenize_and_tensorize(self, expression):
        """İfadeyi ajanın anlayacağı sayısal tensörlere dönüştürür."""
        s_expr = str(expression).replace('**', '^')
        for op in ['+', '-', '*', '/', '(', ')', '^', '=']:
            s_expr = s_expr.replace(op, f' {op} ')
        tokens = s_expr.split()
        indices = [self._VOCAB.get(t, self._VOCAB.get('<UNK>', 0)) for t in tokens]
        
        if len(indices) < self.MAX_SEQ_LEN:
            indices += [self._VOCAB.get('<PAD>' , 0)] * (self.MAX_SEQ_LEN - len(indices))
        else:
            indices = indices[:self.MAX_SEQ_LEN]
            
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def reset(self, complexity=1):
        """5.güne göre müfredat sveiyesine göre soru üreitr."""
        self.current_step = 0
        self.visited_states.clear()
        if self.current_expression:
            self.visited_states.add(str(self.current_expression))# eklendi 28.03
            
        self.last_error = None
        self.history= []
        context= 'basic'
        
        if complexity == 1:
            self.current_expression = self._generate_basic_problem()
            context= 'basic'
        elif complexity == 2:
            self.current_expression = self._generate_algebra_problem()
            context= 'algebra'
        else:
            self.current_expression = self._generate_complex_problem(complexity)
            expr_str= str(self.current_expression)
            if 'sin' in expr_str or 'cos' in expr_str or 'tan' in expr_str:
                context = 'trigonometry'
            elif 'Derivative' in expr_str or 'exp' in expr_str or 'log' in expr_str:
                context = 'calculus'
            else:
                context = 'algebra'
        self.validator._start_repl_for_context(context)
        
        return self._tokenize_and_tensorize(str(self.current_expression))

    def step(self, action_index):
        """Ajanın bir hamle yapmasını sağlar."""
        self.current_step += 1
        self.last_error = None
        action_name = ACTION_TOKENS[action_index]
        done = False
        reward = 0.0
        
        if action_name== 'UNDO':
            if len(self.history) >0:
                last_state = self.history.pop()
                self.current_expression = last_state
                next_state = self._tokenize_and_tensorize(self.current_expression)
                return next_state, -0.1, False, 0.0, 0.0
            else:
                next_state= self._tokenize_and_tensorize(self.current_expression)
                return next_state, -1.0, False, 0.0, 0.0
            
        if self.current_expression is not None:
            self.history.append(self.current_expression)
        
        from sympy import count_ops
        old_expr = self.current_expression
        old_complexity = count_ops(old_expr)
        
        is_valid, lean_msg = self.validator.validate_step(self.current_expression, action_name)
        
        if not is_valid:
            self.last_error = f"Lean Onayı Yok: {lean_msg}"
            return self._tokenize_and_tensorize(self.current_expression), -1.0, False, -1.0, 0.0
        
        try:
            #işlem uygulaması
            if action_name == 'DIFFERENTIATE_X':
                self.current_expression = diff(self.current_expression, self.x)
            elif action_name == 'SIMPLIFY':
                self.current_expression = simplify(self.current_expression)
            elif action_name == 'EXPAND':
                self.current_expression = expand(self.current_expression)
            elif action_name == 'FACTOR':
                self.current_expression = factor(self.current_expression)

            new_expr_str = str(self.current_expression)
            new_complexity = count_ops(self.current_expression)
            
            if new_expr_str=='0' and str(self.target_expression) != '0':
                self.last_error = "sıfır tuzagona düşüldü."
                return self._tokenize_and_tensorize(self.current_expression), -5.0, True, -5.0, 0.0
            if new_expr_str in self.visited_states:
                return self._tokenize_and_tensorize(self.current_expression), -2.0, False, -2.0, 0.0

            #ödül  hesabı
            reward = -0.05# zaman maaliyeti adım basına
            
            if action_name != 'TERMINATE':
                diff_complexity = old_complexity - new_complexity
                if diff_complexity > 0:
                    # Sadeleştirme ödülü
                    reward += 0.2 * np.log1p(float(diff_complexity))
                elif diff_complexity < 0:
                    reward -= 0.05
                else:
                    reward -= 0.3 # Hiçbir şey değişmediyse ceza
            
            if self.is_solved():
                reward = 20.0 - (self.current_step * 0.2) # Hızlı çözene daha çok ödül 15 ti.
                done = True
            elif action_name == 'TERMINATE':
                reward = -5.0 # Yanlış yerde pes etme cezası
                done = True
            elif self.current_step >= self.max_steps:
                reward = -2.0 # Zaman aşımı
                done = True

            return self._tokenize_and_tensorize(self.current_expression), reward, done, reward, 0.0

        except Exception as e:
            self.last_error = f"Eylem hatası: {str(e)}"
            error_reward = -5.0
            done = True
            error_state = torch.tensor([[self._VOCAB.get('<UNK>', 0)]], dtype=torch.long)

            return error_state, error_reward, done, error_reward, 0.0
        
    def is_solved(self):
        """Mevcut ifadenin hedefe eşit olup olmadığını kontrol eder."""
        try:
            return simplify(self.current_expression - self.target_expression) == 0
        except:
            return False
    def get_state(self):
        """mevcut duurm ram i yormaz"""
        return {
            'expr': str(self.current_expression),
            'target': str(self.target_expression),
            'step': self.current_step,
            'prob_str': self.problem_string
        }
    def set_state(self, state_dict):
        """paketi açar ve ortamı ana geri döndürür"""
        from sympy import sympify
        expr_str= state_dict['expr'].replace('=', '==')
        self.current_expression = sympify(expr_str)
        self.target_expression = sympify(state_dict['target'])
        self.current_step = state_dict['step']
        self.problem_string = state_dict['prob_str']
    
    def simulate_step(self, action_index):
        """Ortamı değiştirmeden bir hamlenin sonucunu tahmin eder (Pruning Guard için)."""
        action_name = ACTION_TOKENS[action_index]
        try:
            if action_name == 'DIFFERENTIATE_X':
                return diff(self.current_expression, self.x)
            elif action_name == 'SIMPLIFY':
                return simplify(self.current_expression)
            elif action_name == 'EXPAND':
                return expand(self.current_expression)
            elif action_name == 'FACTOR':
                return factor(self.current_expression)
            return self.current_expression
        except:
            return self.current_expression