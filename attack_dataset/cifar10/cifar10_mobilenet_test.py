import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision.models import mobilenet_v2
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = "cifar10_mobilenet_v2_cpu.pt"
CIFAR_ROOT = "./cifar10_data"

# Change to None to test all 10,000 CIFAR-10 test images.
MAX_TEST_SAMPLES = 1000

OUTPUT_CSV = "mobilenet_v2_cifar10_test_results.csv"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TEST CONFIGURATION")
print("=" * 70)
print("Device:", device)
print("Model path:", MODEL_PATH)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. CIFAR-10 preprocessing
# ============================================================

# This must match the preprocessing used during training.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ]
)


# ============================================================
# 4. Load CIFAR-10 test dataset
# ============================================================

test_dataset = torchvision.datasets.CIFAR10(
    root=CIFAR_ROOT,
    train=False,
    download=True,
    transform=transform,
)

if MAX_TEST_SAMPLES is None:
    number_of_samples = len(test_dataset)
else:
    number_of_samples = min(
        MAX_TEST_SAMPLES,
        len(test_dataset),
    )

print("Number of test images:", number_of_samples)


# ============================================================
# 5. Rebuild MobileNetV2 architecture
# ============================================================

# Do not download pretrained weights during testing.
# The saved state dictionary already contains the trained weights.
model = mobilenet_v2(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    len(CLASS_NAMES),
)


# ============================================================
# 6. Load saved trained weights
# ============================================================

print("\nLoading saved model...")

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    state_dict,
    strict=True,
)

model = model.to(device)
model.eval()

print("Saved model loaded successfully.")


# ============================================================
# 7. Warm-up
# ============================================================

warmup_image, _ = test_dataset[0]

warmup_image = warmup_image.unsqueeze(0).to(device)

with torch.inference_mode():
    _ = model(warmup_image)

if device.type == "cuda":
    torch.cuda.synchronize()

print("Warm-up completed.")


# ============================================================
# 8. Test one image at a time
# ============================================================

results = []

true_label_ids = []
predicted_label_ids = []
latencies = []

print("\n" + "=" * 70)
print("TESTING ONE IMAGE AT A TIME")
print("=" * 70)

with torch.inference_mode():

    for test_index in tqdm(
        range(number_of_samples),
        desc="Testing",
    ):
        image, true_label_id = test_dataset[test_index]

        # Add batch dimension:
        # [3, 224, 224] -> [1, 3, 224, 224]
        image = image.unsqueeze(0).to(device)

        true_label_id = int(true_label_id)
        true_label = CLASS_NAMES[true_label_id]

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        outputs = model(image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start_time

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predicted_label_tensor = torch.max(
            probabilities,
            dim=1,
        )

        predicted_label_id = int(
            predicted_label_tensor.item()
        )

        confidence = float(
            confidence.item()
        )

        predicted_label = CLASS_NAMES[
            predicted_label_id
        ]

        correct = (
            predicted_label_id == true_label_id
        )

        true_label_ids.append(true_label_id)
        predicted_label_ids.append(
            predicted_label_id
        )
        latencies.append(latency)

        results.append(
            {
                "test_index": test_index,
                "true_label_id": true_label_id,
                "true_label": true_label,
                "predicted_label_id": predicted_label_id,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": correct,
                "latency_seconds": latency,
            }
        )

        print(
            f"\nImage {test_index + 1}/{number_of_samples}"
            f"\nTrue label:      {true_label}"
            f"\nPrediction:      {predicted_label}"
            f"\nConfidence:      {confidence:.4f}"
            f"\nCorrect:         {correct}"
            f"\nInference time:  {latency:.6f} seconds"
        )


# ============================================================
# 9. Calculate metrics
# ============================================================

accuracy = accuracy_score(
    true_label_ids,
    predicted_label_ids,
)

macro_f1 = f1_score(
    true_label_ids,
    predicted_label_ids,
    average="macro",
    zero_division=0,
)

average_latency = float(
    np.mean(latencies)
)

median_latency = float(
    np.median(latencies)
)

minimum_latency = float(
    np.min(latencies)
)

maximum_latency = float(
    np.max(latencies)
)

confusion = confusion_matrix(
    true_label_ids,
    predicted_label_ids,
    labels=list(range(10)),
)


# ============================================================
# 10. Display final results
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test images:       {number_of_samples}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Average latency:   {average_latency:.6f} seconds")
print(f"Median latency:    {median_latency:.6f} seconds")
print(f"Minimum latency:   {minimum_latency:.6f} seconds")
print(f"Maximum latency:   {maximum_latency:.6f} seconds")

print("\nClassification report:\n")

print(
    classification_report(
        true_label_ids,
        predicted_label_ids,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(confusion)


# ============================================================
# 11. Save individual test results
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "test_index",
            "true_label_id",
            "true_label",
            "predicted_label_id",
            "predicted_label",
            "confidence",
            "correct",
            "latency_seconds",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:")
print(OUTPUT_CSV)