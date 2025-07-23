import torch
import torch.nn as nn

from bytelatent.test_blt import create_args
from bytelatent.model.blt import ByteLatentTransformer


class BLTEncoder(nn.Module):
    def __init__(self, byte_patch_size=32, embed_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        self.byte_patch_size = byte_patch_size
        self.patch_embedding = nn.Linear(byte_patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.__model = ByteLatentTransformer(create_args(embed_dim))

    def forward(self, tokens: torch.Tensor, batch_size: int=128):
        outputs = []
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i: i + batch_size]
            out = self.__model.forward(batch)
            outputs.append(out)
        return torch.cat(outputs, dim=0)
