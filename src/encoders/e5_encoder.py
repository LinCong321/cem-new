import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class E5Encoder(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        model_name="intfloat/e5-base",
        pooling="mean"
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.pooling = pooling

    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)

        if self.pooling == "mean":
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embeddings = (outputs.last_hidden_state * attention_mask).sum(1)
            sum_mask = attention_mask.sum(1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return embeddings
