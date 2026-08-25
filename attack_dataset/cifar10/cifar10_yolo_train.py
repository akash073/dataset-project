from ultralytics import YOLO

# Load your custom YOLO26 Nano model
model = YOLO('yolo26n-cls.yaml')  # or path to your custom configuration

# Train purely on CPU
results = model.train(data='cifar10', epochs=10, device='cpu')

# Manually save the trained weights
model.save('yolo26n_cifar10_cpu.pt')