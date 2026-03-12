"""
Cross-Modal Fusion
==================
Implements the core architectural innovation: language guided visual attention.

Mathematical formulation:
    The cross-attention mechanism computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    where:
        Q = W_q · lang_tokens    (B, L, d_model)  ← from DistilBERT
        K = W_k · visual_tokens  (B, N, d_model)  ← from EfficientNet
        V = W_v · visual_tokens  (B, N, d_model)  ← from EfficientNet

This forces the model to ask: "given this instruction, which image patches are 
most relevant?" — grounding language semantics in spatial visual evidence.

After cross-attention, a mean-pool over the query dimension yields a single
fused embedding per sample that blends both modalities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossModalFusion(nn.Module):
    """
    Fuses visual tokens (Keys, Values) with language tokens (Queries) via
    multi-head cross-attention, followed by a 2-layer feed-forward network.

    Following the Transformer convention (Vaswani et al., 2017), we apply
    Pre-LayerNorm (Pre-LN) before each sub-layer for more stable training gradients.

    Args:
        d_model: Embedding dimension (must match vision + language encoders).
        nhead:   Number of attention heads.  d_model must be divisible by nhead.
        ffn_dim: Hidden dimension of the position-wise FFN.  Typically 4 × d_model.
        dropout: Applied to attention weights and FFN activations.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead

        # Pre-LN cross-attention: Q from lang, K/V from visual
        self.norm_lang = nn.LayerNorm(d_model)
        self.norm_vis = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,   # (B, seq, d) convention throughout this codebase
        )

        # Pre-LN position-wise FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Self-attention on visual tokens (optional — enriches spatial context
        # before cross-attention queries it)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,        # (B, N, d_model)
        lang_tokens: torch.Tensor,           # (B, L, d_model)
        lang_padding_mask: torch.Tensor,     # (B, L) — True at padding positions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:          (B, L, d_model) — language queries enriched by visual context.
            attn_weights:   (B, nhead, L, N) — attention weight maps (for visualisation).
        """
        # 1. Self-attention on visual tokens to enrich spatial context
        vis_ln = self.norm_self(visual_tokens)
        vis_sa, _ = self.self_attn(vis_ln, vis_ln, vis_ln)
        visual_tokens = visual_tokens + vis_sa

        # 2. Cross-attention: lang queries attend over visual keys/values
        #    Pre-LN applied to both inputs before attention
        lang_ln = self.norm_lang(lang_tokens)
        vis_ln = self.norm_vis(visual_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=lang_ln,
            key=vis_ln,
            value=vis_ln,
            key_padding_mask=None,       # visual tokens are never padded
            need_weights=True,
            average_attn_weights=False,  # return per-head weights (B, nhead, L, N)
        )

        # Residual connection
        fused = lang_tokens + attn_out

        # 3. FFN sub-layer (Pre-LN)
        fused = fused + self.ffn(self.norm_ffn(fused))

        return fused, attn_weights

    @staticmethod
    def mean_pool(fused: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool over the sequence dimension, excluding padding positions.
        Args:
            fused:        (B, L, d_model)
            padding_mask: (B, L) — True at padding positions
        Returns:
            pooled: (B, d_model)
        """
        # Valid token mask: (B, L, 1), float
        valid = (~padding_mask).unsqueeze(-1).float()         # (B, L, 1)
        pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)
        return pooled  # (B, d_model)
