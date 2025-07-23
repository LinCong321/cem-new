import torch
import torch.nn as nn

from bytelatent.test_blt import create_args
from bytelatent.model.blt import ByteLatentTransformer


class BLTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.__model = ByteLatentTransformer(create_args())

    def forward(self, tokens: torch.Tensor, batch_size: int=128):
        outputs = []
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i: i + batch_size]
            out = self.__model.forward(batch)
            outputs.append(out)
        return torch.cat(outputs, dim=0)
