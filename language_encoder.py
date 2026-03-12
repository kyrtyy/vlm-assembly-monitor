"""
Language Encoder
================
Encodes natural-language assembly instructions into dense semantic vectors.
Model: DistilBERT (distilbert-base-uncased) — 66M params, 2x faster than BERT-base,
<3% accuracy delta on most NLU benchmarks (Sanh et al., 2019).

Output tokens from the last hidden state are projected into the same d_model space
as the visual tokens, enabling cross-attention between modalities.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


class LanguageEncoder(nn.Module):
    """
    Tokenises instruction strings and encodes them into d_model-dimensional
    language tokens suitable for cross-modal attention with visual features.

    For a batch of B strings:
        tokens: (B, L, d_model)  — L ≤ max_length (default 64)
        mask:   (B, L)           — True where token is valid
    """

    BERT_DIM = 768  # DistilBERT hidden size

    def __init__(
        self,
        d_model: int = 512,
        max_length: int = 64,
        freeze_bert: bool = False,
        dropout: float = 0.1,
        pretrained_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_name)
        self.bert = DistilBertModel.from_pretrained(pretrained_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

        # Project 768 → d_model
        self.proj = nn.Sequential(
            nn.Linear(self.BERT_DIM, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def tokenize(self, instructions: list[str], device: torch.device) -> dict:
        """
        Convenience method: tokenise a list of instruction strings.
        Returns a dict with 'input_ids' and 'attention_mask' on the target device.
        """
        enc = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, L) — token ids from DistilBertTokenizerFast.
            attention_mask: (B, L) — 1 for real tokens, 0 for padding.
        Returns:
            lang_tokens:  (B, L, d_model) — projected language representations.
            padding_mask: (B, L)          — True where position is PADDING
                                            (inverted from attention_mask).
                                            Used by nn.MultiheadAttention as key_padding_mask.
        """
        # DistilBERT forward — last_hidden_state: (B, L, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Project to shared embedding space
        lang_tokens = self.proj(hidden)  # (B, L, d_model)

        # nn.MultiheadAttention expects padding mask = True where IGNORED
        padding_mask = attention_mask == 0  # (B, L), True at pad positions

        return lang_tokens, padding_mask
