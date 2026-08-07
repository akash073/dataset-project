import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNISTVLMTrainDataset(Dataset):
    def __init__(self, train=True, root="./data"):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        return image, label


class TinyImageEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, text_hidden_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, text_hidden_dim)
        self.proj = nn.Linear(text_hidden_dim, embed_dim)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = x.mean(dim=1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class TinyVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, text_hidden_dim=64):
        super().__init__()
        self.image_encoder = TinyImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TinyTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, token_ids):
        return self.text_encoder(token_ids)

    def forward(self, images, token_ids):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)

        scale = self.logit_scale.exp()
        logits = scale * image_features @ text_features.T
        return logits


class VLMTextProcessor:
    def __init__(self):
        self.token_to_id = {
            "<pad>": 0,
            "digit": 1,
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.seq_len = 2

    def label_to_prompt_tokens(self, label):
        return ["digit", str(label)]

    def prompt_tokens_to_ids(self, tokens):
        ids = [self.token_to_id[t] for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def build_label_token_ids(self, label):
        tokens = self.label_to_prompt_tokens(label)
        return self.prompt_tokens_to_ids(tokens)

    def build_all_class_token_ids(self):
        all_ids = []
        for label in range(10):
            all_ids.append(self.build_label_token_ids(label))
        return torch.stack(all_ids, dim=0)