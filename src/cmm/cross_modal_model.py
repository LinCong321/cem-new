import torch
import torch.nn as nn
from encoders.e5_encoder import E5Encoder
from encoders.blt_encoder import BLTEncoder


class CrossModalModel(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = E5Encoder().to(self.device)
        self.blt_encoder = BLTEncoder().to(self.device)

    def forward(
        self, texts: list[str], tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.to(self.device)
        text_embeds = self.text_encoder(texts)  # [B, D]
        zip_embeds = self.blt_encoder(tokens)   # [B, D]
        return text_embeds, zip_embeds
