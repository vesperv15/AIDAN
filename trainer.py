import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import pickle
import sys
from datetime import datetime
from lean_validator import LeanValidator
from curriculum_manager import CurriculumManager
from memory import SolutionMemory
from aidan_logger import aidan_log
from interfaces import IMathAgent
from config import GAMMA, LR, ENTROPY_BETA, VALUE_LOSS_BETA, EMBED_DIM, FOCUS_DIM, \
                EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE_FREQ, \
                INTRINSIC_BETA, PREDICTION_LOSS_BETA, PREDICTION_DIM
from environment.math_env import ACTION_TOKENS


class RLTrainer:
    def __init__(self, agent:IMathAgent, env, device):
        self.device = device
        self.agent = agent.to(self.device)
        self.env = env
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LR, weight_decay=1e-5)
        self.cirriculum= CurriculumManager()
        self.lean_validator= LeanValidator()
        self.memory = SolutionMemory(FOCUS_DIM, self.device)
        self.model_path = os.environ.get('MODEL_PATH', 'model_data/agent.pth')
        self.memory_path = os.environ.get('MEMORY_PATH', 'model_data/memory.pkl')
        self.log_path= 'model_data/aidan_training.log'
        self.target_agent = self.agent.clone_for_target()
        self.target_agent.eval()
        
        self.epsilon = EPSILON_START
        self.episode_counter = 0
        self.action_space_size = len(ACTION_TOKENS)
        
        self._load_checkpoint()
        
    def _write_log(self,message):
        """Eski log mekanizması, artık merkezi aidan_log'a yönlendirildi."""
        aidan_log.info(message)
            
    def _calculate_returns(self, rewards):
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + GAMMA * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def _calculate_epsilon(self):
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                    np.exp(-self.episode_counter / EPSILON_DECAY)
        return self.epsilon
        
    def _update_target_network(self):
        if self.episode_counter % TARGET_UPDATE_FREQ == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_agent.eval()
            
    def _get_action(self, state_tensor, focus_vector):
        from mcts_logic import mcts_search
        
        if random.random() < self.epsilon:
            action_index = random.randrange(self.action_space_size)
            with torch.no_grad():
                _, value_estimate, prediction_output = self.agent(state_tensor, focus_vector)
        else:
            action_index = mcts_search(self.env, self.agent, self.memory,self.lean_validator)
            with torch.no_grad():
                _, value_estimate, prediction_output = self.agent(state_tensor, focus_vector)
        
        action_tensor = torch.tensor([[action_index]], device=self.device)
        return action_index, action_tensor, None, value_estimate, prediction_output
    
        
    def _load_checkpoint(self):
        """modeli ve vector bankası iceren iskelet hafısaını yukler"""
        try:
            if os.path.exists(self.model_path):
                self.agent.load_state_dict(torch.load(self.model_path, map_location=self.device))
                aidan_log.info(f"[Trainer] Model ağırlıkları ({self.device}) başarıyla yüklendi.")
            
            if os.path.exists(self.memory_path):
                with open(self.memory_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                if isinstance(checkpoint_data, dict):
                    self.memory.success_trajectories= checkpoint_data.get('trajectories', [])
                    self.memory.skeleton_bank= checkpoint_data.get('skeleton_bank',{})
                    aidan_log.info(f"[Trainer] Hafıza yüklendi: {len(self.memory.skeleton_bank)} iskelet yapısı biliniyor.")
                else:
                    # Eski format desteği
                    self.memory.success_trajectories = checkpoint_data
                    aidan_log.info("[Trainer] Eski format hafıza yüklendi.")
        except Exception as e:
            aidan_log.error(f"[Trainer] Yükleme hatası: {e}")
            self.memory.success_trajectories = []
            
    def _save_checkpoint(self):
        """modeli ve iskeletleştirilmis vector bankasını diske kaydeder."""
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.agent.state_dict(), self.model_path)
        try:
            #hem basasrılı onlanları hem de iskelt bankasını paketliyoruz
            checkpoint_data={
                'trajectories': self.memory.success_trajectories,
                'skeleton_bank': self.memory.skeleton_bank
            }
            with open(self.memory_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            aidan_log.info("[Trainer] Model ve Hafıza diske kaydedildi.")
        except Exception as e:
            aidan_log.error(f"[Trainer] Kaydetme hatası: {e}")
            
    def train_episode(self):
        self.episode_counter += 1
        self._calculate_epsilon()
        self._update_target_network()
        constraints= self.cirriculum.get_current_constraints()
        state_tensor = self.env.reset(complexity=constraints["complexity"]).to(self.device)
        
        done = False
        states, rewards, actions, values = [], [], [], [] #hata gelirse 2 tanesi silinecek. geldi sildik.22.03
        target_predictions, prediction_outputs = [], []
        trajectory = []
        total_reward = 0
        step_count=0
        last_milestone_idx=0#hrl makro aksiyon takibi için
        seen_states= set()
        focus_vector =None
        reward_ext= 0.0 ## bakılacak 28.03 unlocal hatası içindi
        
        while not done:
            step_count += 1
            # Burada tensor değil sympy ifadesinin iskeleti üzerinden alıyoruz.
            focus_vector = self.memory.get_focus_vector(self.env.current_expression, self.device) ## hata alırsak curr_expr_str olarak güncellerim
            
            action_index, action_tensor, _, value_estimate, prediction_output = \
                self._get_action(state_tensor, focus_vector)
            
            prev_value= value_estimate.view(-1)[0].item()
            context= self.env.get_context_type()
            tactic= ACTION_TOKENS[action_index]
            curr_expr_str= str(self.env.current_expression)
            if prev_value < -3.5: #3.5 ten çekildi 28.03
                action_index=random.randint(0, len(ACTION_TOKENS)-1)
                tactic= ACTION_TOKENS[action_index]
                self._write_log(f"[Filtre] '{tactic}' yerine RASTGELE hamle seçildi. (Merak Katmanı)")
                is_valid, error_msg = self.lean_validator.validate_step(curr_expr_str, tactic)
            else:
                is_valid, error_msg = self.lean_validator.validate_step(curr_expr_str, tactic)

            if not is_valid:
                # Mantık hatası: Ağır ceza ver ve bölümü bitir (Hata yapmaması için)
                reward_ext = -1.0 #28.03 5 ten 1 e çekildi.
                done = True
                self._write_log(f"MANTIK HATASI | Hamle: {tactic} | Sebep: {error_msg} ")
                next_state_tensor = state_tensor
                next_expr_str= curr_expr_str
                
            else:
                next_state_tuple = self.env.step(action_index)
                next_state_tensor = next_state_tuple[0].to(self.device)
                reward_ext = next_state_tuple[1]
                done = next_state_tuple[2]
                next_expr_str= str(self.env.current_expression)
                
                if next_expr_str not in seen_states and focus_vector is not None and focus_vector.sum().item() != 0:
                    reward_ext += 0.3
                    self._write_log("KÖPRÜ KULLANILDI | Farklı bir konudan transfer ödülü eklendi.")
            
            if self.env.last_error:
                log_msg= f"Bölüm {self.episode_counter} | İfade: {self.env.current_expression} | Hamle: {ACTION_TOKENS[action_index]} | HATA: {self.env.last_error}"
                self._write_log(log_msg)
            
            #next_state_tensor = next_state_tuple[0].to(self.device) if isinstance(next_state_tuple, tuple) else next_state_tuple.to(self.device)
            #reward_ext = next_state_tuple[1] if isinstance(next_state_tuple, tuple) else 0.0
            #done = next_state_tuple[2] if isinstance(next_state_tuple, tuple) else True
            #çökme riskine yorum satırı yapıldı sor ve geri aç lazımsa yoksa sil.
            with torch.no_grad():
                _, next_value_tensor, _ = self.agent(next_state_tensor, focus_vector)
            next_value = next_value_tensor.view(-1)[0].item()#-1 0 kısmı eklendi view ile

            critique_penalty = 0.0
            repetition_penalty = 0.0 #bu eklendi 27.03
            novelty_bonus = 0.0#bu 
            if not done:
                if tactic != 'UNDO':
                # Eğer yeni durum eskisinden ciddi oranda daha kötüyse ceza kes
                    if next_value < prev_value - 0.3:
                        critique_penalty = -1.1
                        self._write_log(f"Self-Critique: '{tactic}' hamlesi durumu kötüleştirdi. V: {prev_value:.2f} -> {next_value:.2f}")
                if next_expr_str in seen_states:
                    repetition_penalty = -5.0#burası belli eklemdi 27.03
                    self._write_log(f"Repetition: '{tactic}' hamlesiyle daha önce görülen duruma dönüldü.")
                else:
                    seen_states.add(next_expr_str)
                
                if hasattr(self.memory, 'skeleton_bank'):#aynı şekilde 27.03
                    next_skeleton = self.env.get_skeleton(self.env.current_expression)
                    if next_skeleton not in self.memory.skeleton_bank:
                        novelty_bonus = 1.0
                        
            with torch.no_grad():
                next_state_emb_mean = self.agent.embedding(next_state_tensor).mean(dim=1).detach()
                target_prediction_vec = self.agent.inverse_embedding(next_state_emb_mean)

            intrinsic_loss_scalar = F.mse_loss(prediction_output, target_prediction_vec).detach() 
            reward_int = INTRINSIC_BETA * (-intrinsic_loss_scalar.item())
            
            total_reward_step = reward_ext + reward_int + critique_penalty + repetition_penalty + novelty_bonus

            states.append(state_tensor)
            rewards.append(total_reward_step)
            values.append(value_estimate)
            prediction_outputs.append(prediction_output)
            target_predictions.append(target_prediction_vec)
            
            actions.append(action_tensor)#ekledik 18.03
            #dummy kısmı ve entropies tamamen silindi.18.03
            
            embedding_tensor = self.agent.embedding(state_tensor).detach().squeeze()
            trajectory.append(embedding_tensor)
            
            if tactic in ['SIMPLIFY', 'FACTOR', 'EXPAND'] and not self.env.last_error:
                steps_taken = len(states) - last_milestone_idx
                
                if steps_taken >2:
                    flush_reward= sum(rewards[last_milestone_idx:])
                    
                    del states[last_milestone_idx + 1 : -1]
                    del actions[last_milestone_idx + 1 : -1]
                    del values[last_milestone_idx + 1 : -1]
                    del prediction_outputs[last_milestone_idx + 1 : -1]
                    del target_predictions[last_milestone_idx + 1 : -1]
                    del rewards[last_milestone_idx + 1 : -1]
                    del trajectory[last_milestone_idx + 1 : -1]
                    
                    rewards[-1]=flush_reward
                    self._write_log(f"h-rl context flush: {steps_taken} adımlık boşluk sıkıştırıldı.gürültü silindi.")
                last_milestone_idx = len(states) - 1
            
            state_tensor = next_state_tensor
            total_reward += total_reward_step
#burası. 19.03 halledildi
        total_loss_tensor = self._perform_optimization(states, rewards, actions, values, prediction_outputs, target_predictions)

        total_loss_val = total_loss_tensor.item() if isinstance(total_loss_tensor, torch.Tensor) else 0.0

        avg_reward = total_reward / len(rewards) if rewards else 0
        log_msg = f"bölüm:{self.episode_counter} / ödül:{total_reward:.2f} / kayıp:{total_loss_val:.4f} / hata: {self.env.last_error}"
        self._write_log(log_msg)
        
        is_solved= self.env.is_solved()
        self.cirriculum.report_result(is_solved)
        if is_solved:
            self.memory.add_solution(trajectory, self.env.current_expression)

        level_up= self.cirriculum.report_result(is_solved)
        if level_up:
            self._write_log(f"seviye atlandı: {self.cirriculum.current_level} ---")
        
        try:
            del states, rewards, actions, values
            del target_predictions, prediction_outputs
            del trajectory
        except NameError:
            pass
        import gc
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        #garbage collection kısmı.
        aidan_log.debug(f"[Trainer] Bölüm {self.episode_counter} sonrası RAM ve VRAM temizlendi.")
        #self._perform_optimization(states,rewards,actions, values, prediction_outputs, target_predictions) ,,bunu sil dio?.18.03 23.38
        
        return total_reward,avg_reward, total_loss_val, len(self.memory.skeleton_bank)
    #burası sorrr 18.03. 19.03 halledildi
    
    def _perform_optimization(self, states,rewards, actions, values, pred_out, target_pred):
        """kayıp fonksiyonu hesaplamaa ve ağırlık güncelleme"""
        returns= self._calculate_returns(rewards).to(self.device)
        if len(returns)== 0: return

        states_tensor = torch.cat(states, dim=0).to(self.device)
        actions_tensor= torch.cat(actions, dim=0).to(self.device)
        prediction_outputs_batch = torch.cat(pred_out, dim=0).to(self.device)
        target_predictions_batch = torch.cat(target_pred, dim=0).to(self.device)
        #28.03 komle silinip burası eklendi
        raw_focus = self.memory.get_focus_vector(self.env.current_expression, self.device)
        batch_focus = raw_focus[0:1, :].expand(states_tensor.size(0), -1)
        #
        policy_logits, values_batch, _ = self.agent(states_tensor,batch_focus)
        values_batch= values_batch.view(-1) #düzeltildi 22.03
        returns= returns.view(-1)#eklendi 22.03
        
        policy_probs= F.softmax(policy_logits, dim=-1)
        log_policy_probs= F.log_softmax(policy_logits,dim=-1)
        
        log_probs_batch = log_policy_probs.gather(1, actions_tensor.view(-1, 1)).squeeze()
        entropies_mean = -(policy_probs * log_policy_probs).sum(dim=-1).mean()
        
        advantage = returns - values_batch.detach()
        
        policy_loss = -(log_probs_batch * advantage).mean()
        value_loss = F.mse_loss(values_batch, returns)
        prediction_loss = F.mse_loss(prediction_outputs_batch, target_predictions_batch)

        total_loss = policy_loss + \
                     VALUE_LOSS_BETA * value_loss - \
                     ENTROPY_BETA * entropies_mean + \
                     PREDICTION_LOSS_BETA * prediction_loss

        # 4. AĞIRLIKLARI GÜNCELLE
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss
    def run_training(self, NUM_EPISODES):
        start_msg= f"Eğitim Başlatılıyor ({self.device})..."
        print(f"\n{start_msg}")
        aidan_log.info(start_msg)
        
        for episode in range(self.episode_counter, NUM_EPISODES):
            total_reward, avg_reward, total_loss, memory_size = self.train_episode()
            
            if (episode + 1) % 10 == 0:
                log_msg= f"Bölüm: {episode+1}/{NUM_EPISODES} | T.Ödül: {total_reward:.2f} | Kayıp: {total_loss:.4f} | Hafıza: {memory_size}"
                print(log_msg)
                aidan_log.info(log_msg)

            if (episode + 1) % 200 == 0:
                self._save_checkpoint()
        
        self._save_checkpoint()
        finish_msg = "✅ Eğitim Tamamlandı."
        print(f"\n{finish_msg}")
        aidan_log.info(finish_msg)
        
        if hasattr(self.lean_validator, 'close_repl'):
            self.lean_validator.close_repl()
            aidan_log.info("[trainer] lean repl güvenlice kapatıldı.")