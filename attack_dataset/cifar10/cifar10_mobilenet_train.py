import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def main():
    # 1. Define CPU Device explicitly
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preprocessing for MobileNetV2
    # MobileNetV2 expects 224x224 images, so we resize CIFAR-10's 32x32 images.
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Using a smaller batch size (e.g., 16 or 32) so your CPU doesn't run out of memory
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # 3. Load Pretrained MobileNetV2 and Modify the Head
    print("Initializing MobileNetV2...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Replace the final classification layer (originally 1000 outputs) with 10 outputs for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 10)
    model = model.to(device)

    # 4. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop (Running 1 epoch as an example due to CPU speeds)
    print("Starting training on CPU... (This may take some time)")
    model.train()
    
    for epoch in range(10):  # Adjust epochs as needed
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Training finished.")

    # Ensure the model is explicitly on the CPU before saving
    model.to('cpu')

    # Save the state dictionary with the .pt extension
    model_save_path = "mobilenet_v2_cpu.pt"
    torch.save(model.state_dict(), model_save_path)

    print(f"Model successfully saved to {model_save_path}")
    print(f"Model successfully saved to {model_save_path}")

if __name__ == '__main__':
    main()