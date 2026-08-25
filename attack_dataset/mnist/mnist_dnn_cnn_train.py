# -*- coding: utf-8 -*-
"""train_cifar10.ipynb

CIFAR-10 training for SimpleCNN and SimpleDNN
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker

from models.model import SimpleCNN, SimpleDNN


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# CIFAR-10 TRAINING DATA
# ============================================================

train_loader = DataLoader(

    datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transform
    ),

    batch_size=64,

    shuffle=True
)


# ============================================================
# TRAINING ENGINE
# ============================================================

def run_experiment(model_class, name):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = model_class().to(
        device
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # CodeCarbon
    # --------------------------------------------------------

    tracker = EmissionsTracker(

        project_name=
            f"CIFAR10_Fingerprint_{name}",

        output_dir="."
    )


    tracker.start()


    print(
        f"\nStarting {name} "
        f"CIFAR-10 Training..."
    )


    # ========================================================
    # Training
    # ========================================================

    model.train()


    for epoch in range(2):

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


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

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
            f"Epoch [{epoch + 1}/2] "
            f"Loss: {average_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )


    # ========================================================
    # Stop energy tracking
    # ========================================================

    tracker.stop()


    # ========================================================
    # Save model
    # ========================================================

    save_path = (
        f"cifar10_"
        f"{name.lower()}.pth"
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
# RUN BOTH MODELS
# ============================================================

run_experiment(
    SimpleCNN,
    "CNN"
)


run_experiment(
    SimpleDNN,
    "DNN"
)