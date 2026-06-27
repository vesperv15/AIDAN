from abc import ABC, abstractmethod
import torch

class IMathAgent(ABC):
    """
    Sistem 1 (Sezgisel Ajan) Kontratı.
    MCTS veya Trainer, içeride Transformer mı yoksa LSTM mi olduğunu bilmez.
    Sadece bu metodların var olduğuna güvenir.
    """
    @abstractmethod
    def forward(self, state_tensor: torch.Tensor, focus_vector: torch.Tensor):
        """Durum ve odak vektörünü alıp (policy_logits, value, prediction) dönmek zorundadır."""
        pass

    @abstractmethod
    def clone_for_target(self):
        """Hedef ağ güncellemeleri için kendi kopyasını dönmek zorundadır."""
        pass
class IValidator(ABC):
    """
    Sistem 2 Doğrulayıcı (Lean 4) Kontratı.
    Çevre (Environment) Lean'in nasıl çalıştığını (PIPE, REPL) umursamaz.
    Sadece 'Bu hamle geçerli mi?' diye sorar.
    """
    @abstractmethod
    def validate_step(self, current_expr: str, action_key: str):
        """Matematiksel ifadeyi ve eylemi alıp (is_valid: bool, mesaj: str) dönmek zorundadır."""
        pass