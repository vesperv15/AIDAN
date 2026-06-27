import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MAX_STEPS, NUM_EPISODES
from environment.math_env import MathEnv, ACTION_TOKENS
from agent_model import RnnMathAgent
from trainer import RLTrainer

def main():
    """Ana eğitim fonksiyonu."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    env = MathEnv(max_steps=MAX_STEPS)
    agent = RnnMathAgent(vocab_size=env.vocab_size).to(device)
    trainer = RLTrainer(agent, env, device)

    try:
        if isinstance(NUM_EPISODES, str):
            final_episodes= int(NUM_EPISODES)
        else:
            final_episodes=NUM_EPISODES
        
        trainer.run_training(final_episodes)
    except KeyboardInterrupt:
        print("\nEğitim kullanıcı tarafından durduruldu. Kaydediliyor...")
        trainer._save_checkpoint()
    print("program sonlandı")

if __name__ == "__main__":
    main()