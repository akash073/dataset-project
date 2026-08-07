import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTTokenDataset(Dataset):
    def __init__(self, train=True, root="./data"):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

        self.token_to_id = {
            "<pad>": 0,
            "<bos>": 1,
            "classify_digit": 2,
            "<ans>": 3,
        }

        for i in range(16):
            self.token_to_id[f"P{i}"] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # <bos>, classify_digit, 49 patch tokens, <ans>
        self.seq_len = 52
        self.vocab_size = len(self.token_to_id)

    def image_to_patch_tokens(self, image_tensor):
        # image_tensor: [1, 28, 28]
        image = image_tensor.squeeze(0)
        tokens = []

        for i in range(0, 28, 4):
            for j in range(0, 28, 4):
                patch = image[i:i + 4, j:j + 4]
                avg_val = patch.mean().item()  # 0 to 1
                bin_id = min(int(avg_val * 16), 15)
                tokens.append(f"P{bin_id}")

        return tokens

    def build_input_ids(self, image_tensor):
        patch_tokens = self.image_to_patch_tokens(image_tensor)
        sequence = ["<bos>", "classify_digit"] + patch_tokens + ["<ans>"]
        input_ids = [self.token_to_id[token] for token in sequence]
        return torch.tensor(input_ids, dtype=torch.long)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        input_ids = self.build_input_ids(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_ids, label_tensor


class TinyLLMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        num_classes=10,
        dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.transformer(x)

        # use final token representation
        x = self.norm(x[:, -1, :])
        logits = self.classifier(x)
        return logits