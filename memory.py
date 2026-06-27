import torch
import random
import numpy as np
import re
from sympy import Symbol, srepr
try:
    from config import FOCUS_DIM
except ImportError:
    FOCUS_DIM = 32

class SolutionMemory:
    """
    gelişmiş vector bank başarılı çözümleri yapısal iskeletlerine göre saklar ve ram kullanımını optimize eder.
    """
    def __init__(self, focus_dim, device):
        self.focus_dim = focus_dim
        self.device = device
        self.skeleton_bank= {}
        self.success_trajectories = []
        self.max_bank_size= 50000 # 15 gb ram koruması.
        self.episode_counter = 0#silmelimiyim bakıcam.16.03.26
        self.epsilon = 1.0#silmelimyim bakıcam

    def _abstract_expression(self, expr_str):
        """ Gizli Köprüler: Matematiksel bir ifadeyi yapısal iskeletine dönüştürür.
        Örnek: 'sin(x)^2 + cos(x)^2 = 1' -> 'FUNC(VAR)^2 + FUNC(VAR)^2 = 1'
        Örnek: 'log(y)' -> 'FUNC(VAR)' """
        
        expr= str(expr_str)
        expr= re.sub(r'\b[a-zA-Z]\b', 'VAR', expr)
        func_patterns = ['sin', 'cos', 'tan', 'cot', 'log', 'ln', 'exp']
        for func in func_patterns:
            expr = expr.replace(func, 'FUNC')
            
        # sayıları NUM yap (İsteğe bağlı, ajanın 2 ile 5'i aynı yapı görmesi için)
        expr = re.sub(r'\b\d+\b', 'NUM', expr)
            
        return expr
    
    def get_skeleton(self, expr):
        """sympy ifadesini değişkenlerden bağımsız bir şekilde iskelete donustur"""
        try:
            symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
            skeleton_expr = expr
            for i, sym in enumerate(symbols):
                skeleton_expr = skeleton_expr.subs(sym, Symbol(f'VAR{i}'))
            return srepr(skeleton_expr)
        except Exception:
            return str(expr)
    
    def add_solution(self, trajectory, final_expression):
        """
        Başarılı bir çözüm yörüngesini hafızaya ekler.
        iskelet anahatrıyla ekler.
        """
        self.success_trajectories.append(trajectory)
        
        #gizli köprü ifadeyi soyutla.
        skeleton = self._abstract_expression(final_expression)
        
        # yapı dana önce öğrenilmediyse bankaya ekle vector.
        if skeleton not in self.skeleton_bank:
            if trajectory and len(trajectory) > 0:
                # ispat sonu en rafine kökü sakla
                self.skeleton_bank[skeleton] = trajectory[-1].detach()
        
        # ram için en eskiyi sil.
        if len(self.skeleton_bank) > self.max_bank_size:
            first_key = next(iter(self.skeleton_bank))
            del self.skeleton_bank[first_key]

    def get_focus_vector(self, current_expression, device):
        """
        mevcut ifadenin iskeletine göre hafızadan benzer tecrübe arar.
        """
        skeleton= self._abstract_expression(current_expression)
        
        if skeleton in self.skeleton_bank:
            focus_vector=self.skeleton_bank[skeleton]
            if focus_vector.dim() ==1:
                focus_vector=focus_vector.unsqueeze(0)
            return focus_vector.detach().to(device)
            
        if not self.success_trajectories:
            return torch.zeros(1, self.focus_dim).to(device)
        
        try:
            random_trajectory = random.choice(self.success_trajectories)

            if not random_trajectory or len(random_trajectory) ==0:
                return torch.zeros(1, self.focus_dim).to(device)

            raw_vector = random_trajectory[-1]

            if isinstance(raw_vector, (list, tuple)):
                raw_vector = raw_vector[0]

            # Tensör kontrolü ve Güvenli Dönüş
            if isinstance(raw_vector, torch.Tensor):
                focus_vector = raw_vector.clone().detach()
                if focus_vector.dim() == 1:
                    focus_vector = focus_vector.unsqueeze(0)
                return focus_vector.to(device)
                
        except Exception:
            # ayıklama hatasında sistem çökmemesi için
            pass

        # yoksa sıfır vectoru don
        return torch.zeros(1, self.focus_dim).to(device)#garbage collection için detach eklendi.