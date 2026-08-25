# -*- coding: utf-8 -*-
"""train_cifar10_rgb.py

CIFAR-10 training for SimpleCNN and SimpleDNN
using native RGB 3x32x32 images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker


# ============================================================
# CNN FOR CIFAR-10 RGB
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # CIFAR-10 has 3 RGB channels
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Input: 32x32
        # conv1 -> 30x30
        # pool  -> 15x15
        # conv2 -> 13x13
        # pool  -> 6x6
        self.fc1 = nn.Linear(
            64 * 6 * 6,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv2(x)
            )
        )

        x = torch.flatten(
            x,
            1
        )

        x = F.relu(
            self.fc1(x)
        )

        return self.fc2(x)


# ============================================================
# DNN FOR CIFAR-10 RGB
# ============================================================

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        # CIFAR-10:
        # 3 x 32 x 32 = 3072 features
        self.fc1 = nn.Linear(
            3 * 32 * 32,
            512
        )

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.fc3 = nn.Linear(
            256,
            128
        )

        self.fc4 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = torch.flatten(
            x,
            1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = F.relu(
            self.fc3(x)
        )

        return self.fc4(x)


# ============================================================
# CIFAR-10 RGB PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(
            0.4914,
            0.4822,
            0.4465
        ),
        std=(
            0.2470,
            0.2435,
            0.2616
        )
    )
])


# ============================================================
# CIFAR-10 TRAIN DATA
# ============================================================

train_dataset = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# ============================================================
# VERIFY INPUT SHAPE
# ============================================================

images, labels = next(
    iter(train_loader)
)

print(
    "Input shape:",
    images.shape
)

# Expected:
# [64, 3, 32, 32]

assert images.shape[1:] == (
    3,
    32,
    32
)


# ============================================================
# TRAINING ENGINE
# ============================================================

def run_experiment(
    model_class,
    name
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )


    model = model_class().to(
        device
    )


    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        32,
        32
    ).to(device)


    with torch.no_grad():

        dummy_output = model(
            dummy_input
        )


    print(
        f"{name} output shape:",
        dummy_output.shape
    )


    # Must be 10 classes
    assert dummy_output.shape == (
        2,
        10
    )


    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    criterion = nn.CrossEntropyLoss()


    tracker = EmissionsTracker(
        project_name=
            f"CIFAR10_RGB_{name}",
        output_dir=".",
        log_level="error"
    )


    tracker.start()


    print(
        f"\nStarting {name} "
        f"CIFAR-10 Training..."
    )


    model.train()


    NUM_EPOCHS = 2


    for epoch in range(
        NUM_EPOCHS
    ):

        running_loss = 0.0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
            )


            predicted = torch.argmax(
                outputs,
                dim=1
            )


            total += labels.size(0)


            correct += (
                predicted
                ==
                labels
            ).sum().item()


        epoch_accuracy = (
            100.0
            *
            correct
            /
            total
        )


        average_loss = (
            running_loss
            /
            len(train_loader)
        )


        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )


    emissions = tracker.stop()


    print(
        f"{name} emissions:",
        emissions
    )


    save_path = (
        f"cifar10_{name.lower()}.pth"
    )


    torch.save(
        model.state_dict(),
        save_path
    )


    print(
        f"{name} Complete."
    )


    print(
        f"Weights saved to: "
        f"{save_path}"
    )


# ============================================================
# TRAIN CNN
# ============================================================

run_experiment(
    SimpleCNN,
    "CNN"
)


# ============================================================
# TRAIN DNN
# ============================================================

run_experiment(
    SimpleDNN,
    "DNN"
)