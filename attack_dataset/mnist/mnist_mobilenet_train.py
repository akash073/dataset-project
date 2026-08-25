import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


def main():

    # ============================================================
    # 1. Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )


    # ============================================================
    # 2. MNIST preprocessing for MobileNetV2
    # ============================================================
    #
    # Original MNIST:
    #     1 x 28 x 28
    #
    # MobileNetV2 expects:
    #     3 x 224 x 224
    #
    # Therefore:
    #     grayscale -> RGB-like 3 channels
    #     28x28 -> 224x224
    #
    # Since we use pretrained ImageNet weights, use ImageNet
    # normalization.
    # ============================================================

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.485,
                0.456,
                0.406
            ),
            std=(
                0.229,
                0.224,
                0.225
            )
        )
    ])


    # ============================================================
    # 3. Load MNIST training dataset
    # ============================================================

    print(
        "Loading MNIST dataset..."
    )


    trainset = (
        torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    )


    trainloader = (
        torch.utils.data.DataLoader(
            trainset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
    )


    # ============================================================
    # Verify input dimensions
    # ============================================================

    sample_images, sample_labels = next(
        iter(trainloader)
    )


    print(
        "Input batch shape:",
        sample_images.shape
    )

    print(
        "Label batch shape:",
        sample_labels.shape
    )


    # Expected:
    # [32, 3, 224, 224]
    assert sample_images.shape[1:] == (
        3,
        224,
        224
    )


    # ============================================================
    # 4. Load pretrained MobileNetV2
    # ============================================================

    print(
        "Initializing MobileNetV2..."
    )


    model = mobilenet_v2(
        weights=
            MobileNet_V2_Weights.DEFAULT
    )


    # ============================================================
    # Replace final classifier
    # ============================================================
    #
    # Original ImageNet:
    #     1000 classes
    #
    # MNIST:
    #     10 classes
    # ============================================================

    in_features = (
        model.classifier[1]
        .in_features
    )


    model.classifier[1] = (
        nn.Linear(
            in_features,
            10
        )
    )


    model = model.to(
        device
    )


    # ============================================================
    # 5. Loss + optimizer
    # ============================================================

    criterion = (
        nn.CrossEntropyLoss()
    )


    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # ============================================================
    # 6. Training
    # ============================================================

    print(
        "Starting MNIST training..."
    )


    model.train()


    NUM_EPOCHS = 1


    for epoch in range(
        NUM_EPOCHS
    ):

        running_loss = 0.0

        correct = 0

        total = 0


        for i, (
            inputs,
            labels
        ) in enumerate(
            trainloader
        ):

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )


            # --------------------------------------------
            # Reset gradients
            # --------------------------------------------

            optimizer.zero_grad()


            # --------------------------------------------
            # Forward
            # --------------------------------------------

            outputs = model(
                inputs
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
            # Optimize
            # --------------------------------------------

            optimizer.step()


            running_loss += (
                loss.item()
            )


            # --------------------------------------------
            # Training accuracy
            # --------------------------------------------

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


            # --------------------------------------------
            # Print progress
            # --------------------------------------------

            if i % 100 == 99:

                batch_accuracy = (
                    100.0
                    *
                    correct
                    /
                    total
                )


                print(

                    f"[Epoch {epoch + 1}, "
                    f"Batch {i + 1}] "

                    f"Loss: "
                    f"{running_loss / 100:.3f} "

                    f"Accuracy: "
                    f"{batch_accuracy:.2f}%"

                )


                running_loss = 0.0


        # ========================================================
        # Epoch accuracy
        # ========================================================

        epoch_accuracy = (

            100.0
            *
            correct
            /
            total
        )


        print(
            f"Epoch {epoch + 1} "
            f"Training Accuracy: "
            f"{epoch_accuracy:.2f}%"
        )


    print(
        "Training finished."
    )


    # ============================================================
    # 7. Move model to CPU before saving
    # ============================================================

    model.to(
        "cpu"
    )


    # ============================================================
    # 8. Save state dictionary
    # ============================================================

    model_save_path = (
        "mobilenet_v2_mnist_cpu.pt"
    )


    torch.save(
        model.state_dict(),
        model_save_path
    )


    print(
        f"Model successfully saved to "
        f"{model_save_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    main()