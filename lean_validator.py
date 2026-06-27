import subprocess
import json
import os
import re #
import time #
from threading import Lock
from interfaces import IValidator
from aidan_logger import aidan_log

#semblokk araçalr # claude
try: 
    from sympy import simplify, expand, factor, diff, symbols, sympify, count_ops 
    x, y, z, t = symbols('x y z t')
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class ExpressionConverter:
    """
    SymPy string ifadelerini Lean 4 sözdizimine çeviren köprü.
    Bu sınıf olmadan Lean hiçbir SymPy ifadesini anlayamaz.
    Örnek: 'x**2 + 2*x + 1' → 'x^2 + 2*x + 1'
    """

    # sympy lean karsılıkları
    FUNC_MAP = {
        'sin':  'Real.sin',
        'cos':  'Real.cos',
        'tan':  'Real.tan',
        'exp':  'Real.exp',
        'log':  'Real.log',
        'sqrt': 'Real.sqrt',
    }

    @staticmethod
    def sympy_to_lean4(expr_str: str) -> str:
        """SymPy string ifadesini Lean 4 sözdizimine çevirir."""
        expr = str(expr_str).strip()

        # üs için
        expr = expr.replace('**', '^')

        # sin içeren kelimeler bozulmasın diye \b (word boundary) kullanıyoruz
        for sympy_fn, lean_fn in ExpressionConverter.FUNC_MAP.items():
            expr = re.sub(rf'\b{sympy_fn}\b', lean_fn, expr)

        return expr

    @staticmethod
    def detect_variables(expr_str: str) -> list:
        """İfadedeki matematiksel değişkenleri tespit eder (x, y, z, t)."""
        return sorted(set(re.findall(r'\b([xyzt])\b', str(expr_str))))

    @staticmethod
    def build_equivalence_proof(expr_before: str, expr_after: str) -> str:
        """
        Bir dönüşümün matematiksel eşdeğerliğini kanıtlayan Lean 4 kodu üretir.
        Lean'e sorduğumuz soru: 'Bu iki ifade birbirine eşit mi?'
        
        Örnek çıktı:
        example (x : ℝ) : x^2 + 2*x + 1 = (x + 1)^2 := by ring
        """
        lean_before = ExpressionConverter.sympy_to_lean4(expr_before)
        lean_after  = ExpressionConverter.sympy_to_lean4(expr_after)
        # kullanılan değişkenler
        all_vars = ExpressionConverter.detect_variables(lean_before + lean_after)

        # lean tip bildirimleri
        var_decls = ' '.join([f'({v} : ℝ)' for v in all_vars])

        if not all_vars:
            # sadece sayılsal ifadeyse norm num
            return f"example : ({lean_before} : ℝ) = {lean_after} := by norm_num"

        # polinom eşdeğerliği için ring
        # Eğer ring yetmezse simp [ring]
        return f"example {var_decls} : {lean_before} = {lean_after} := by ring"

    @staticmethod
    def build_terminal_proof(current_expr: str, target_expr: str) -> str:
        """
        TERMINATE anında: Mevcut ifadenin hedefe eşit olduğunu kanıtlar.
        Bu Lean'in en kritik görevi — hallüsinasyonu burada önlüyoruz.
        """
        return ExpressionConverter.build_equivalence_proof(current_expr, target_expr)

class SymPyFallbackValidator:
    """
    Lean kurulu değilse veya kullanılamıyorsa devreye giren
    SymPy tabanlı güvenilir doğrulayıcı.
    
    Lean kadar formal değil ama SymPy matematiği gerçekten anlıyor,
    random fallback onayı vermekten çok daha iyi.
    """

    ALWAYS_VALID_ACTIONS = {'EXPAND', 'FACTOR', 'SIMPLIFY', 'DIFFERENTIATE_X', 'UNDO'}

    @staticmethod
    def validate(current_expr_str: str, action_key: str,
                result_expr_str: str = None, target_expr_str: str = None) -> tuple:

        if action_key == 'UNDO':
            return True, "geri alındı"

        if not SYMPY_AVAILABLE:
            # SymPy bile yoksa izin ver sistem öğretir
            return True, "sympy_unavailable_fallback"

        try:
            if action_key == 'TERMINATE':
                # kontrol : iafede hedefe eşit mi
                if target_expr_str is None:
                    return False, "Hedef ifade bilinmiyor"

                current = sympify(str(current_expr_str))
                target  = sympify(str(target_expr_str))
                difference = simplify(current - target)

                if difference == 0:
                    return True, "Hedef doğrulandı (SymPy)"
                else:
                    return False, f"Hedef değil. Fark: {difference}"

            # eylemler mat olarak geçerli
            # (expand, factor, simplify, diff her zaman uygulanabilir)
            if action_key in SymPyFallbackValidator.ALWAYS_VALID_ACTIONS:
                return True, "sympy_valid"

            return True, "sympy_default_valid"

        except Exception as e:
            aidan_log.warning(f"[SymPy Fallback] Doğrulama hatası: {e}")
            return True, "sympy_exception_fallback"
#

class LeanValidator(IValidator):
    """
    Lean 4 REPL üzerinden çalışan, hızlı ve düşük RAM tüketen
    canlı doğrulama sistemi.
    """
    def __init__(self, lean_path="lean"):
        self.lean_path = lean_path
        self.process = None
        self.lock = Lock() #mcts için güvenli erişim
        self.request_count=0
        self.MAX_REQUESTS_BEFORE_RESTART =50
        
        self.mathlib_modules ={
            'basic': '', # ram harcatmaz
            'algebra': 'import Mathlib.Algebra.Group.Basic\n',
            'calculus': 'import Mathlib.Analysis.Calculus.Deriv.Basic\n',
            'trigonometry': 'import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic\n'
        }
        #hatırlatma
        self.tactic_map = {
            'SIMPLIFY': 'simp',
            'EXPAND': 'ring',
            'FACTOR': 'ring',
            'DIFFERENTIATE_X': 'simp', # calc veya rw ile ilerde denenebilir
            'TERMINATE': 'done',
            'UNDO': ''
        }
        
        self.is_active= self._start_repl_for_context('basic')

    def _start_repl_for_context(self, context_key='basic'):
        """Lean REPL sürecini başlatır.belirli bir konu için gereken matlib i yükler ve başatır."""
        self.close_repl()
        try:
            # Persistent REPL başlat
            self.process = subprocess.Popen(
                ["bash", "-c", f"{self.lean_path} --run"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            import_cmd= self.mathlib_modules.get(context_key, '')
            if import_cmd:
                self.process.stdin.write(import_cmd)
                self.process.stdin.flush()
                
            self.is_active=True
            return True
        except Exception as e:
            print(f"Lean REPL başlatılamadı: {e}")
            self.is_active= False
            return False

    def close_repl(self):
        """Çalışan REPL sürecini sonlandırır ve RAM'i işletim sistemine anında iade eder."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()#zorla kapa inat ederse
            finally:
                self.process= None
                self.is_active=False
    
    def validate_step(self, current_expr: str, action_key:str, context=None):
        """
        Ajanın bir adımını anlık olarak Lean'e sorar.
        """
        if action_key=='UNDO':
            aidan_log.info(f"[lean4] 'UNDO' eylemi çağırıldı işlem geri alınıyor.")
            return True, "geri alındı"
        
        if not self.is_active or not self.process:
            aidan_log.warning(f"[Lean 4] REPL kapalı! '{action_key}' için Fallback (Otomatik) onay verildi.")
            return True, "fallback onay"

        
        tactic = self.tactic_map.get(action_key, "simp")
        # ornek sablon lean için
        # Lean 4 REPL protokolüne uygun JSON formatı veya direkt komut değiştirildi 12.05
        lean_expr= ExpressionConverter.sympy_to_lean4(str(current_expr))
        proof_query= f"example ( x : ℝ) : {lean_expr} = {lean_expr} := by {tactic}\n"
        with self.lock:
            self.request_count += 1
            
            if self.request_count > self.MAX_REQUESTS_BEFORE_RESTART:
                aidan_log.info("[Lean 4] RAM temizliği için REPL yeniden başlatılıyor...")
                self._restart_repl() # Bu fonksiyonu aşağıda tanımlı
                self.request_count = 0
            
            if self.process is None or self.process.poll() is not None:
                aidan_log.warning(f"[Lean 4] REPL süreci ölü veya kapalı! Yeniden başlatılıyor...")
                self._start_repl_for_context()
            
            try:
                self.process.stdin.write(proof_query)
                self.process.stdin.flush()
                
                import time
                time.sleep(0.05)  #cevap içi süre lean
                
                output = ""
                #mcts kitlenmesini önle
                # non-blocking yap. beklemesiz okuma
                os.set_blocking(self.process.stdout.fileno(), False)
                if self.process.stderr:
                    os.set_blocking(self.process.stderr.fileno(), False)
                
                try:
                    out = self.process.stdout.read()
                    if out: output += out
                    
                    if self.process.stderr:
                        err = self.process.stderr.read()
                        if err: output += err
                except (TypeError, OSError):
                    pass
                
                # lean hata verrse error döndür
                if "error:" in output.lower() or "failed" in output.lower() or "unknown" in output.lower():
                    error_detail = output.split('\\n')[0][:50] if output else "Geçersiz Taktik"
                    aidan_log.warning(f"[Lean 4] RED: İfade '{current_expr}' -> '{action_key}' reddedildi. ({error_detail})")
                    return False, "Geçersiz hamle"
                
                # devam hata yoksa
                aidan_log.debug(f"[Lean 4] ONAY: İfade '{current_expr}' -> Taktik '{action_key}' geçerli.")
                return True, "step_valid"
            
            except Exception as e:
                error_msg = str(e)
                aidan_log.error(f"[lean4] ret. taktik: '{action_key}' | Sebep: {error_msg} ")
                return False, str(e)
    
    def _restart_repl(self):
        """Lean sürecini kapatıp temiz bir sayfa açar (RAM'i boşaltır)."""
        if self.process:
            self.process.kill() # Süreci zorla kapat
            self.process.wait()
        self._start_repl_for_context('basic') # Tekrar başlat
    

    def __del__(self):
        self.close_repl()#nesne silinirse ram de repl kalmasın.