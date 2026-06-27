import torch
import os

def fix():
    path = "model_data/agent.pth"
    
    if os.path.exists(path):
        print(f"✅ Model dosyası bulundu: {path}")
        try:
            state_dict = torch.load(path, map_location="cpu")
            weight_key = 'inverse_embedding.weight'
            bias_key = 'inverse_embedding.bias'
            
            if weight_key in state_dict:
                print(f"🛠️ Mevcut boyut: {state_dict[weight_key].shape}")
                print("🔄 Katman 128-padding sistemine göre yeniden boyutlandırılıyor...")
                
                old_weight= state_dict[weight_key]
                new_weight= torch.randn(50,32) * 0.01 #başlangıç gürültüsü.
                
                #eski ağırlıkların sığdığı kadarını yeni matrise kopyala
                min_rows = min(old_weight.shape[0], new_weight.shape[0])
                min_cols = min(old_weight.shape[1], new_weight.shape[1])
                new_weight[:min_rows, :min_cols] = old_weight[:min_rows, :min_cols]
                
                state_dict[weight_key] = new_weight
                state_dict[bias_key] = torch.zeros(50)
                
                torch.save(state_dict, path)
                print("✨ BAŞARILI: agent.pth artık yeni sisteme uyumlu!")
            else:
                print("⚠️ Katman bulunamadı, model yapısı farklı olabilir.")
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")
    else:
        print(f"❌ HATA: '{path}' konumunda dosya bulunamadı!")
        print("Lütfen Docker içindeki klasör yapısını kontrol edin.")

if __name__ == "__main__":
    fix()