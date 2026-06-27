import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCUS_DIM, EMBED_DIM, HIDDEN_DIM, ACTION_TOKENS, PREDICTION_DIM
from interfaces import IMathAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Transformer'lar LSTM gibi sırayı otomatik bilemezler.
    İfadedeki karakterlerin pozisyonunu modele öğretmek için bu modül şarttır.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class RnnMathAgent(nn.Module, IMathAgent):
    def __init__(self, vocab_size,embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, **kwargs):
        super(RnnMathAgent, self).__init__()
        self.vocab_size = vocab_size
        self.action_space_size = len(ACTION_TOKENS)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        # nano transformer
        #giriş boyutu= 32 (Embed) + 32 (Focus) = 64
        self.d_model = embedding_dim + FOCUS_DIM
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        # köprüler
        self.bridge = nn.Linear(self.d_model, hidden_dim)
        # --------------------------------
        
        self.policy_head = nn.Linear(hidden_dim, self.action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.prediction_head = nn.Linear(hidden_dim, PREDICTION_DIM)
        self.inverse_embedding = nn.Linear(embedding_dim, PREDICTION_DIM) #64 yap diyor şekil hatası?.
        self.to(device)

    def forward(self, state_tensor, focus_vector):
        state_tensor = state_tensor.to(device)
        focus_vector = focus_vector.to(device)
        
        embedded = self.embedding(state_tensor) # (batch, seq, 32)
        
        if focus_vector.dim() == 2:
            focus_expanded = focus_vector.unsqueeze(1).expand(-1, embedded.size(1), -1)
        else:
            focus_expanded = focus_vector
        #22.03 eklendi.
        B_emb, seq_emb, _ = embedded.shape
        _, seq_foc, _ = focus_expanded.shape
        
        #en büyük Batch ve seq uzunluklarını bul
        max_seq = max(seq_emb, seq_foc)
        #İki matrisi de bu maksimum boyutlara esnetiyoruz
        # matris o boyuttaysa expand bozmaz.
        embedded_expanded = embedded.expand(B_emb, max_seq, -1)
        if focus_expanded.size(0) != B_emb:
            focus_expanded = focus_expanded.mean(dim=0, keepdim=True).expand(B_emb, max_seq, -1)
        else:
            focus_expanded = focus_expanded.expand(B_emb, max_seq, -1)
        #27.03 değiştirildi batch 128 hatası sorunu.
        combined_input = torch.cat([embedded_expanded, focus_expanded], dim=-1) # (batch, seq, 64)
        
        #tranformer akısı
        # opsiyonal kodlama
        x = self.pos_encoder(combined_input)
        #dikkat kısmı
        transformed_out = self.transformer_encoder(x) # (batch, seq, 64)
        
        # global avarage pooling.
        pooled_out = transformed_out.mean(dim=1) # (batch, 64)
        #eski boyuta (128) genişlet
        hidden = F.leaky_relu(self.bridge(pooled_out), negative_slope=0.01) # (batch, 128)
        # -------------------------
        
        return self.policy_head(hidden), self.value_head(hidden), self.prediction_head(hidden)

    def clone_for_target(self):
        clone = RnnMathAgent(vocab_size=self.vocab_size, action_dim=self.action_space_size)
        clone.load_state_dict(self.state_dict())
        clone.to(device)
        clone.eval()
        return clone