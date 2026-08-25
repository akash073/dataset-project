import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# CNN MODEL
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
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

        self.fc1 = nn.Linear(
            64 * 5 * 5,
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
# DNN MODEL
# ============================================================

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(
            28 * 28,
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
# MNIST PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


# ============================================================
# MNIST TRAINING DATA
# ============================================================

train_dataset = datasets.MNIST(
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
        f"\nUsing device: {device}"
    )


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
    # Number of epochs
    # --------------------------------------------------------

    NUM_EPOCHS = 10


    # ========================================================
    # Training
    # ========================================================

    print(
        f"\nStarting {name} MNIST training..."
    )


    model.train()


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


            # --------------------------------------------
            # Clear gradients
            # --------------------------------------------

            optimizer.zero_grad()


            # --------------------------------------------
            # Forward
            # --------------------------------------------

            outputs = model(
                images
            )


            # --------------------------------------------
            # Loss
            # --------------------------------------------

            loss = criterion(
                outputs,
                labels
            )


            # --------------------------------------------
            # Backward
            # --------------------------------------------

            loss.backward()


            # --------------------------------------------
            # Update weights
            # --------------------------------------------

            optimizer.step()


            # --------------------------------------------
            # Statistics
            # --------------------------------------------

            running_loss += (
                loss.item()
            )


            predicted = torch.argmax(
                outputs,
                dim=1
            )


            total += (
                labels.size(0)
            )


            correct += (
                predicted
                ==
                labels
            ).sum().item()


        # ====================================================
        # Epoch metrics
        # ====================================================

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


    # ========================================================
    # Save model
    # ========================================================

    save_path = (
        f"mnist_"
        f"{name.lower()}.pth"
    )


    torch.save(
        model.state_dict(),
        save_path
    )


    print(
        f"\n{name} training complete."
    )


    print(
        f"Weights saved to: "
        f"{save_path}"
    )


# ============================================================
# RUN CNN
# ============================================================

run_experiment(
    SimpleCNN,
    "CNN"
)


# ============================================================
# RUN DNN
# ============================================================

run_experiment(
    SimpleDNN,
    "DNN"
)