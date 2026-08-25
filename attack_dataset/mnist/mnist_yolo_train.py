
from ultralytics import YOLO



# from pathlib import Path

# from torchvision.datasets import MNIST
# from tqdm import tqdm


# OUTPUT_ROOT = Path("../data")

# TRAIN_DIR = OUTPUT_ROOT / "train"
# VAL_DIR = OUTPUT_ROOT / "val"


# # ============================================================
# # Create class folders
# # ============================================================

# for split_dir in [TRAIN_DIR, VAL_DIR]:
#     for digit in range(10):
#         (split_dir / str(digit)).mkdir(
#             parents=True,
#             exist_ok=True
#         )


# # ============================================================
# # Download/load MNIST
# # ============================================================

# train_dataset = MNIST(
#     root="./mnist_raw",
#     train=True,
#     download=True
# )

# test_dataset = MNIST(
#     root="./mnist_raw",
#     train=False,
#     download=True
# )


# # ============================================================
# # Save training images
# # ============================================================

# print("Converting MNIST training data...")

# for i, (image, label) in enumerate(
#     tqdm(train_dataset)
# ):

#     save_path = (
#         TRAIN_DIR
#         / str(label)
#         / f"{i:06d}.png"
#     )

#     image.save(save_path)


# # ============================================================
# # Save validation/test images
# # ============================================================

# print("Converting MNIST validation data...")

# for i, (image, label) in enumerate(
#     tqdm(test_dataset)
# ):

#     save_path = (
#         VAL_DIR
#         / str(label)
#         / f"{i:06d}.png"
#     )

#     image.save(save_path)


# print("\nDone.")

# print("Train:", TRAIN_DIR.resolve())
# print("Val  :", VAL_DIR.resolve())

# Load YOLO classification model
model = YOLO("yolo26n-cls.yaml")

# Train on MNIST using CPU
results = model.train(
    data="../data",
    epochs=10,
    device="cpu"
)

# Save trained model
model.save("yolo26n_mnist_cpu.pt")