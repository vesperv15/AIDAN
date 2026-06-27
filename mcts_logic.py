import math
import torch
import numpy as np
import random
from environment.math_env import MathEnv, ACTION_TOKENS
from aidan_logger import aidan_log

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        """Düğümün ortalama değerini (beklenen ödül) döndürür."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

def mcts_search(env, agent, memory, lean_mgr, success_rate=0.5, num_simulations=None):
    """
    4.gün lean onaylı yaratcılık derin arama
    """
    initial_state = env.get_state()
    initial_expr = env.current_expression
    root= MCTSNode(initial_state)
    context = env.get_context_type()
    expr_complexity= len(str(initial_expr))
    max_complexity_limit= len(str(initial_expr)) * 2
    
    aidan_log.info(f"[MCTS] Yeni arama başlatıldı. İfade: {initial_expr} | Bütçe: {num_simulations} Simülasyon")
    
    if num_simulations is None:
        num_simulations= get_dynamic_simulations(initial_expr)
    dynamic_tau= get_dynamic_threshold(success_rate, expr_complexity)
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        # secim
        while node.children:
            node = max(node.children.values(), key=lambda c: ucb_score(node, c))
            search_path.append(node)
            
        env.set_state(node.state)
        #genişleme
        state_tensor = env._tokenize_and_tensorize(env.current_expression).to(agent.embedding.weight.device)
        focus_vector = memory.get_focus_vector(env.current_expression, state_tensor.device)
        
        with torch.no_grad():
            policy_logits, value_pred, _ = agent(state_tensor, focus_vector)
            probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        
        valid_actions_found=False
        for action_idx, prob in enumerate(probs):
            if prob < dynamic_tau: continue #düşük ihtimalleri ram için ele
            
            tactic= ACTION_TOKENS[action_idx]
            temp_expr= env.simulate_step(action_idx)
            if len(str(temp_expr)) > max_complexity_limit:
                aidan_log.warning(f"[MCTS] Budandı: '{tactic}'. Sebep: Karmaşıklık patlaması ({len(str(temp_expr))} > {max_complexity_limit})")
                continue
            
            if value_pred.item() < -7.0: #2.5 ten 7 yaptık
                aidan_log.warning(f"[MCTS] Budandı: '{tactic}'. Sebep: Çok düşük değer ({value_pred.item():.2f})")
                continue
            
            if lean_mgr.validate_step(env.current_expression, tactic, context): #verify proof füzeltildi
                env.step(action_idx)
                node.children[action_idx] = MCTSNode(env.get_state(), parent=node, action=action_idx, prior=prob)
                env.set_state(node.state)
                valid_actions_found = True
                
        v= value_pred.item() if valid_actions_found else -1.0 #eğer lean onay vermese ölü dal olarak işaretlenir.
        
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += v
    env.set_state(initial_state)
    
    if not root.children:
        return random.randrange(len(ACTION_TOKENS))
    
    return max(root.children.items(), key=lambda i: i[1].visit_count)[0]

def ucb_score(parent, child):
    """
    UCB1: Keşif (Exploration) ve Sömürü (Exploitation) dengesi.
    """
    pb_c= 1.25
    u= pb_c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    
    return child.value() + u
#4.gün buraya kadardı.
def get_dynamic_simulations(expression):
    """
    4. GÜN: RAM verimliliği için ifade karmaşıklığına göre bütçe belirler.
    """
    complexity = len(str(expression))
    if complexity < 15: return 8
    if complexity < 40: return 15
    return 30

def get_dynamic_threshold(success_rate, complexity):
    """
    6. GÜN: Yaratıcılığı Koruyan Dinamik Budama (Pruning).
    Ajan başarısız oluyorsa kapıları açar (tau düşer), RAM doluyorsa kapıları daraltır.
    """
    base_tau = 0.04  # %4 eşiği
    
    #takıldıysa sacma ama özgün yollar için ödül değeri
    if success_rate < 0.2:
        return 0.005
    
    #ifade karmasıksa ram için arttır eşiği
    return base_tau * (1 + (complexity * 0.1))