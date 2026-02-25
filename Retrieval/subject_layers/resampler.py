import torch
import torch.nn as nn

class Resampler(nn.Module):
    """
    Perceiver-style resampler:
    x: [B, L, D] -> z_tokens: [B, K, D]
    """
    def __init__(self, d_model: int, num_queries: int = 8, num_heads: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)  # [K, D]

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_x = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        q_ln = self.ln_q(q)
        x_ln = self.ln_x(x)

        z, _ = self.attn(query=q_ln, key=x_ln, value=x_ln, need_weights=False)  # [B, K, D]
        z = q + self.drop(z)  # residual
        z = z + self.drop(self.ffn(z))  # residual FFN
        return z
