import torch
import torch.nn as nn
from encoders.e5_encoder import E5Encoder
from encoders.blt_encoder import BLTEncoder


class CrossModalModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.text_encoder = E5Encoder().to(device)
        self.blt_encoder = BLTEncoder().to(device)

    def forward(self, text_list, tokens):
        text_embeds = self.text_encoder(text_list)  # [B, D]
        zip_embeds = self.blt_encoder(tokens)  # [B, D]
        return text_embeds, zip_embeds
