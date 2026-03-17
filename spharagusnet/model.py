"""Arquitectura del modelo de clasificación de bloques OCR."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

if _TORCH_OK:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DocumentTextExtractor(nn.Module):
        """
        Modelo para clasificación de bloques relevantes/irrelevantes.

        Arquitectura:
        - Embedding layer (vocab → hidden_dim vectors)
        - Block classifier (2 capas FC con ReLU + dropout) → {0: irrelevante, 1: relevante}
        - LSTM (reservado para futuras extensiones de generación)
        """

        def __init__(self, vocab_size: int = 10000, hidden_dim: int = 512,
                     device: torch.device | None = None):
            super().__init__()
            self.device = device or DEVICE

            self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.block_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 2),
            )
            # LSTM reservado para compat con checkpoints anteriores
            self.text_generator = nn.LSTM(
                hidden_dim, hidden_dim, num_layers=2, batch_first=True,
            )
            self.output_layer = nn.Linear(hidden_dim, vocab_size)
            self.to(self.device)

        def forward(self, text_tokens: torch.Tensor,
                    block_positions: torch.Tensor | None = None):
            text_tokens = text_tokens.to(self.device)
            embedded = self.text_embedding(text_tokens)
            block_scores = self.block_classifier(embedded.mean(dim=1))
            lstm_out, _ = self.text_generator(embedded)
            output = self.output_layer(lstm_out)
            return output, block_scores
else:
    DEVICE = None  # type: ignore[assignment]
    DocumentTextExtractor = None  # type: ignore[assignment,misc]
