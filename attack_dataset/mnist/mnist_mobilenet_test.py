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

MODEL_PATH = "mobilenet_v2_mnist_cpu.pt"

MNIST_ROOT = "./data"

# None = test all 10,000 MNIST test images
MAX_TEST_SAMPLES = 1000

OUTPUT_CSV = "mobilenet_v2_mnist_test_results.csv"

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("TEST CONFIGURATION")
print("=" * 70)

print("Device:", device)
print("Model path:", MODEL_PATH)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. MNIST preprocessing
# ============================================================
#
# Must match training:
#
# MNIST:
#   1 x 28 x 28
#
# MobileNetV2 input:
#   3 x 224 x 224
#
# ============================================================

transform = transforms.Compose(
    [
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
            ),
        ),
    ]
)


# ============================================================
# 4. Load MNIST test dataset
# ============================================================

test_dataset = torchvision.datasets.MNIST(
    root=MNIST_ROOT,
    train=False,
    download=True,
    transform=transform,
)


if MAX_TEST_SAMPLES is None:

    number_of_samples = len(
        test_dataset
    )

else:

    number_of_samples = min(
        MAX_TEST_SAMPLES,
        len(test_dataset),
    )


print(
    "Number of MNIST test images:",
    number_of_samples
)


# ============================================================
# 5. Rebuild MobileNetV2 architecture
# ============================================================
#
# Do not download pretrained weights here.
#
# The saved state_dict already contains the trained weights.
# ============================================================

model = mobilenet_v2(
    weights=None
)


in_features = (
    model.classifier[1]
    .in_features
)


model.classifier[1] = nn.Linear(
    in_features,
    len(CLASS_NAMES),
)


# ============================================================
# 6. Load saved trained weights
# ============================================================

print(
    "\nLoading saved MNIST model..."
)


state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)


model.load_state_dict(
    state_dict,
    strict=True,
)


model = model.to(
    device
)

model.eval()


print(
    "Saved MobileNetV2 MNIST model "
    "loaded successfully."
)


# ============================================================
# 7. Warm-up
# ============================================================

warmup_image, _ = (
    test_dataset[0]
)


# [3, 224, 224]
# ->
# [1, 3, 224, 224]

warmup_image = (
    warmup_image
    .unsqueeze(0)
    .to(device)
)


with torch.inference_mode():

    _ = model(
        warmup_image
    )


if device.type == "cuda":

    torch.cuda.synchronize()


print(
    "Warm-up completed."
)


# ============================================================
# 8. Test one image at a time
# ============================================================

results = []

true_label_ids = []

predicted_label_ids = []

latencies = []


print(
    "\n" + "=" * 70
)

print(
    "TESTING MNIST ONE IMAGE AT A TIME"
)

print(
    "=" * 70
)


with torch.inference_mode():

    for test_index in tqdm(

        range(
            number_of_samples
        ),

        desc="Testing",

    ):

        # ----------------------------------------------------
        # Load one MNIST image
        # ----------------------------------------------------

        image, true_label_id = (
            test_dataset[
                test_index
            ]
        )


        # [3, 224, 224]
        # ->
        # [1, 3, 224, 224]

        image = (
            image
            .unsqueeze(0)
            .to(device)
        )


        true_label_id = int(
            true_label_id
        )


        true_label = CLASS_NAMES[
            true_label_id
        ]


        # ----------------------------------------------------
        # Synchronize GPU before timing
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()


        # ----------------------------------------------------
        # Start inference timer
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )


        outputs = model(
            image
        )


        # ----------------------------------------------------
        # Synchronize after inference
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()


        latency = (
            time.perf_counter()
            -
            start_time
        )


        # ====================================================
        # Probabilities
        # ====================================================

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )


        confidence, predicted_label_tensor = (
            torch.max(
                probabilities,
                dim=1,
            )
        )


        predicted_label_id = int(
            predicted_label_tensor
            .item()
        )


        confidence = float(
            confidence.item()
        )


        predicted_label = CLASS_NAMES[
            predicted_label_id
        ]


        correct = (
            predicted_label_id
            ==
            true_label_id
        )


        # ====================================================
        # Store metrics
        # ====================================================

        true_label_ids.append(
            true_label_id
        )


        predicted_label_ids.append(
            predicted_label_id
        )


        latencies.append(
            latency
        )


        results.append(
            {
                "test_index":
                    test_index,

                "true_label_id":
                    true_label_id,

                "true_label":
                    true_label,

                "predicted_label_id":
                    predicted_label_id,

                "predicted_label":
                    predicted_label,

                "confidence":
                    confidence,

                "correct":
                    correct,

                "latency_seconds":
                    latency,
            }
        )


        print(

            f"\nImage "
            f"{test_index + 1}/"
            f"{number_of_samples}"

            f"\nTrue label:      "
            f"{true_label}"

            f"\nPrediction:      "
            f"{predicted_label}"

            f"\nConfidence:      "
            f"{confidence:.4f}"

            f"\nCorrect:         "
            f"{correct}"

            f"\nInference time:  "
            f"{latency:.6f} seconds"

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
    np.mean(
        latencies
    )
)


median_latency = float(
    np.median(
        latencies
    )
)


minimum_latency = float(
    np.min(
        latencies
    )
)


maximum_latency = float(
    np.max(
        latencies
    )
)


confusion = confusion_matrix(
    true_label_ids,
    predicted_label_ids,
    labels=list(
        range(10)
    ),
)


# ============================================================
# 10. Display final results
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL MNIST TEST RESULTS"
)

print(
    "=" * 70
)


print(
    f"Test images:       "
    f"{number_of_samples}"
)


print(
    f"Accuracy:          "
    f"{accuracy:.4f}"
)


print(
    f"Macro F1:          "
    f"{macro_f1:.4f}"
)


print(
    f"Average latency:   "
    f"{average_latency:.6f} seconds"
)


print(
    f"Median latency:    "
    f"{median_latency:.6f} seconds"
)


print(
    f"Minimum latency:   "
    f"{minimum_latency:.6f} seconds"
)


print(
    f"Maximum latency:   "
    f"{maximum_latency:.6f} seconds"
)


print(
    "\nClassification report:\n"
)


print(
    classification_report(
        true_label_ids,
        predicted_label_ids,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


print(
    "\nConfusion matrix:"
)

print(
    confusion
)


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

    writer.writerows(
        results
    )


print(
    "\nResults saved to:"
)

print(
    OUTPUT_CSV
)