import torch
import torch.nn as nn

from bytelatent.test_blt import create_args
from bytelatent.model.blt import ByteLatentTransformer


class BLTEncoder(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ByteLatentTransformer(create_args())
        self.model.to(self.device)

    def forward(self, tokens: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        tokens = tokens.to(self.device)
        outputs = []

        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i: i + batch_size]
            out = self.model(batch)
            outputs.append(out)

        return torch.cat(outputs, dim=0)
