"""
Temporal Module
===============
Models the sequential nature of assembly tasks across T video frames.

Why a Causal Transformer (not LSTM)?
- Self-attention is fully parallelisable during training — no sequential hidden
  state dependency.
- The causal mask ensures token at position t only attends to positions ≤ t,
  preserving temporal causality (future frames cannot influence past predictions).
- LSTM gradient flow degrades over long sequences; self-attention gradient path 
  length is O(1) regardless of sequence length.

Input:  per-frame fused embeddings  (B, T, d_model) — from CrossModalFusion.mean_pool
Output: contextualised embeddings    (B, T, d_model) — each position aware of its past

The [CLS]-style token pooling strategy aggregates the final output over T into a
single (B, d_model) vector for the output heads when predicting the overall
assembly state. For per-frame box predictions, the full (B, T, d_model) is used.
"""
import math
import torch
import torch.nn as nn


class CausalTemporalTransformer(nn.Module):
    """
    A stack of causally-masked Transformer encoder layers applied over the time
    dimension of a video clip.

    Args:
        d_model:   Embedding dimension.
        nhead:     Number of attention heads.
        num_layers: Depth of the transformer stack.
        ffn_dim:   FFN hidden size.
        max_seq_len: Maximum number of frames (T_max). Used to pre-compute the
                     causal mask and the temporal positional embeddings.
        dropout:   Applied throughout.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 2048,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learned temporal positional embeddings
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Transformer encoder layers (Pre-LN via norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Pre-compute causal mask once; register as buffer (moves with .to(device))
        causal_mask = self._make_causal_mask(max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

    @staticmethod
    def _make_causal_mask(size: int) -> torch.Tensor:
        """
        Upper-triangular mask filled with -inf, zeros on and below the diagonal.
        nn.TransformerEncoder adds this to attention logits, masking future positions.
        Shape: (size, size)
        """
        mask = torch.triu(
            torch.full((size, size), float("-inf")), diagonal=1
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — per-frame fused embeddings.
        Returns:
            out: (B, T, d_model) — temporally contextualised embeddings.
        """
        B, T, D = x.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
            "Re-initialise with a larger max_seq_len."
        )

        # Add temporal positional embeddings
        x = x + self.temporal_pos_embed[:, :T, :]

        # Apply causal mask (slice to current sequence length)
        mask = self.causal_mask[:T, :T]

        out = self.transformer(x, mask=mask, is_causal=True)
        return self.norm(out)


class TemporalAggregator(nn.Module):
    """
    Aggregates the T-frame temporal output into a single clip-level embedding
    for the state classification head.

    Strategy: Weighted mean-pool using learned temperature-scaled attention
    over the T frame positions. This is more informative than simple mean-pool
    because late frames (near task completion) naturally receive higher weight.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        # Scalar attention score per position
        self.attn_score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            clip_embed: (B, d_model) — aggregated clip-level representation.
        """
        # Attention weights over T: (B, T, 1) → softmax → (B, T, 1)
        scores = self.attn_score(x)              # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # (B, T, 1)

        # Weighted sum: (B, T, 1) * (B, T, d) → sum over T → (B, d_model)
        clip_embed = (weights * x).sum(dim=1)
        return clip_embed
