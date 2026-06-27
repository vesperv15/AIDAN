class CurriculumManager:
    def __init__(self):
        self.current_level = 0
        self.success_threshold = 0.8  # %80 başarıda seviye atla
        self.recent_successes = []
        self.window_size = 20 # Son 20 soruyu baz al
        
        # seviye tanımları zorluk vs.
        self.levels = {
            0: {"name": "Temel Aritmetik", "complexity": 1, "tactics": ["rw", "refl"]},
            1: {"name": "Basit Cebir", "complexity": 2, "tactics": ["rw", "refl", "simp", "linarith"]},
            2: {"name": "İleri Cebir & Eşitsizlik", "complexity": 3, "tactics": ["rw", "refl", "simp", "linarith", "ring"]},
            3: {"name": "Olimpiyat Seviyesi", "complexity": 5, "tactics": ["all"]}
        }

    def report_result(self, success: bool):
        self.recent_successes.append(1.0 if success else 0.0)
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
            
        avg_success = sum(self.recent_successes) / len(self.recent_successes)
        
        
        
        # Seviye Atlatma Mantığı
        if avg_success >= self.success_threshold and len(self.recent_successes) == self.window_size:
            if self.current_level < max(self.levels.keys()):
                self.current_level += 1
                self.recent_successes = [] #yeni seivye için sıfırla
                print(f"--> [Curriculum] Başarı yüksek (%{avg_success*100:.0f}), zorluk ARTIRILDI: Seviye {self.current_level}")
                return True # Seviye atladı
            elif avg_success< 0.20:
                if self.current_level > 0:
                    self.current_level -= 1
                    self.recent_successes = []
                    print(f"--> [Curriculum] Başarı çok düşük (%{avg_success*100:.0f}), zorluk AZALTILDI: Seviye {self.current_level}")
                    return True
        return False

    def get_success_rate(self):
        """son window size 20 sorudaki güncel başarı oranını döndürür"""
        if not self.recent_successes:
            return 0.5
        return sum(self.recent_successes) / len(self.recent_successes)
    
    def get_adaptive_max_steps(self, base_steps=20):
        """
        Ajan başarısızlık sarmalına girdiğinde, keşif yapabilmesi için
        ona daha fazla hamle yapma (düşünme) hakkı tanır.
        """
        current_rate = self.get_success_rate()
        
        #basarı azsa %15
        if current_rate < 0.15:
            return base_steps * 2  #adım saıyısı arrttır
            
        #<%35 için 
        elif current_rate < 0.35:
            return int(base_steps * 1.5) #1.5 kat arttır adım saıysını
        
        return base_steps#tammasa devam et.
    
    def get_current_constraints(self):
        return self.levels[self.current_level]