"""
Load Only: YOLO26 Nano MNIST CPU Inference
=============================================

This script DOES NOT train.

It only:
1. Loads yolo26n_mnist_cpu.pt
2. Downloads the MNIST test split
3. Selects exactly 1,000 test samples
4. Runs inference one image at a time on CPU
5. Saves per-sample telemetry and overall model metrics

Install:
    pip install -U ultralytics torch torchvision pandas numpy psutil \
        py-cpuinfo codecarbon scikit-learn tqdm pillow

Run:
    python load_only_test_1000_yolo26n_mnist_cpu.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from ultralytics import YOLO
import ultralytics


# =============================================================================
NUM_TEST_SAMPLES = 10

def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"

CPU_MODEL_NAME = get_cpu_model()
def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(exist_ok=True)
DEVICE_LOG_DIR = OUTPUT_ROOT / f"{DEVICE_SHORT}"
DEVICE_LOG_DIR.mkdir(exist_ok=True)

DATA_ROOT = Path("./mnist_data")
# =============================================================================

SEED = 42

MODEL_PATH = Path.cwd() / "yolo26n_mnist_cpu.pt"



IMAGE_SIZE = 28
DEVICE = "cpu"

ENABLE_CODECARBON = True
FLUSH_EVERY = 25

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

N_CLASSES = len(CLASS_NAMES)

TELEMETRY_CSV_PATH = DEVICE_LOG_DIR / "yolo_test_telemetry_1000.csv"
# PREDICTIONS_CSV_PATH = OUTPUT_ROOT / "test_predictions_1000.csv"
# RESULTS_JSON_PATH = OUTPUT_ROOT / "test_results_1000.json"
# CLASSIFICATION_REPORT_CSV_PATH = (
#     OUTPUT_ROOT / "classification_report_1000.csv"
# )
# CONFUSION_MATRIX_CSV_PATH = (
#     OUTPUT_ROOT / "confusion_matrix_1000.csv"
# )
CODECARBON_CSV_PATH = OUTPUT_ROOT / "codecarbon" / "yolo_codecarbon_inference.csv"
# RETURN_FILES_JSON_PATH = OUTPUT_ROOT / "return_files.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))


# =============================================================================
# OPTIONAL CODECARBON
# =============================================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = codecarbon.__version__
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

def get_cpu_model() -> str:
    try:
        import cpuinfo

        return cpuinfo.get_cpu_info().get(
            "brand_raw",
            "Unknown",
        )
    except Exception:
        return platform.processor() or "Unknown"


def get_os_full_name() -> str:
    system = platform.system()
    architecture = platform.machine()

    if system == "Windows":
        return (
            f"Windows {platform.release()} "
            f"{platform.version()} {architecture}"
        )

    if system == "Linux":
        try:
            os_information: dict[str, str] = {}

            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8",
            ) as file_handle:
                for line in file_handle:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        os_information[key] = value.strip('"')

            return (
                f"{os_information.get('PRETTY_NAME', 'Linux')} "
                f"{architecture}"
            )
        except Exception:
            return (
                f"Linux {platform.release()} "
                f"{architecture}"
            )

    if system == "Darwin":
        return (
            f"macOS {platform.mac_ver()[0]} "
            f"{architecture}"
        )

    return (
        f"{system} {platform.release()} "
        f"{architecture}"
    )


CPU_MODEL_NAME = get_cpu_model()
OS_FULL_NAME = get_os_full_name()

PYTHON_VERSION = sys.version.split()[0]
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = torchvision.__version__
ULTRALYTICS_VERSION = ultralytics.__version__

CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)
SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total / (1024 ** 3),
    2,
)


def make_stable_device_id() -> str:
    raw_value = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]


def get_memory_footprint_mb() -> float | None:
    try:
        process = psutil.Process(os.getpid())

        return round(
            process.memory_info().rss / (1024 * 1024),
            4,
        )
    except Exception:
        return None


def get_cpu_usage() -> float | None:
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_ram_usage() -> float | None:
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return None


def get_cpu_freq() -> float | None:
    try:
        frequency = psutil.cpu_freq()

        return (
            round(frequency.current, 2)
            if frequency is not None
            else None
        )
    except Exception:
        return None


def get_cpu_cores_used() -> int | None:
    try:
        return sum(
            1
            for value in psutil.cpu_percent(percpu=True)
            if value > 1.0
        )
    except Exception:
        return None


# =============================================================================
# MODEL STATISTICS
# =============================================================================

def get_model_statistics(model: YOLO) -> dict[str, int]:
    torch_model = model.model

    total_parameters = sum(
        parameter.numel()
        for parameter in torch_model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in torch_model.parameters()
        if parameter.requires_grad
    )

    convolution_layers = 0
    linear_layers = 0
    normalization_layers = 0
    activation_layers = 0
    pooling_layers = 0

    for module in torch_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            convolution_layers += 1
        elif isinstance(module, torch.nn.Linear):
            linear_layers += 1
        elif isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.LayerNorm,
                torch.nn.GroupNorm,
            ),
        ):
            normalization_layers += 1
        elif isinstance(
            module,
            (
                torch.nn.ReLU,
                torch.nn.LeakyReLU,
                torch.nn.SiLU,
                torch.nn.GELU,
                torch.nn.Sigmoid,
                torch.nn.Tanh,
            ),
        ):
            activation_layers += 1
        elif isinstance(
            module,
            (
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.AdaptiveAvgPool2d,
            ),
        ):
            pooling_layers += 1

    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "module_count": int(len(list(torch_model.modules()))),
        "convolution_layers": convolution_layers,
        "linear_layers": linear_layers,
        "normalization_layers": normalization_layers,
        "activation_layers": activation_layers,
        "pooling_layers": pooling_layers,
    }


# =============================================================================
# HELPERS
# =============================================================================

def append_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return

    new_dataframe = pd.DataFrame(rows)

    if output_path.exists():
        old_dataframe = pd.read_csv(
            output_path,
            on_bad_lines="skip",
        )

        for column in new_dataframe.columns:
            if column not in old_dataframe.columns:
                old_dataframe[column] = None

        for column in old_dataframe.columns:
            if column not in new_dataframe.columns:
                new_dataframe[column] = None

        new_dataframe = new_dataframe[
            old_dataframe.columns
        ]

        final_dataframe = pd.concat(
            [old_dataframe, new_dataframe],
            ignore_index=True,
        )
    else:
        final_dataframe = new_dataframe

    final_dataframe.to_csv(
        output_path,
        index=False,
    )


def prediction_quality(
    probabilities: np.ndarray,
) -> tuple[float, float, float]:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    probabilities = (
        probabilities / probabilities.sum()
    )

    sorted_probabilities = np.sort(
        probabilities
    )[::-1]

    confidence = float(
        sorted_probabilities[0]
    )

    margin = float(
        sorted_probabilities[0]
        - sorted_probabilities[1]
        if len(sorted_probabilities) > 1
        else sorted_probabilities[0]
    )

    entropy = float(
        -np.sum(
            probabilities * np.log(probabilities)
        )
    )

    return (
        round(confidence, 6),
        round(margin, 6),
        round(entropy, 6),
    )


# =============================================================================
# ENERGY TRACKING
# =============================================================================

def run_with_energy_tracking(
    function,
    *args,
    project_name: str,
    output_file: str,
    **kwargs,
) -> dict[str, Any]:
    should_track = (
        ENABLE_CODECARBON
        and CODECARBON_AVAILABLE
    )

    if should_track:
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(OUTPUT_ROOT),
            output_file=output_file,
            log_level="error",
            save_to_file=True,
            measure_power_secs=1,
        )

        tracker.start()
        start_time = time.perf_counter()

        try:
            result = function(*args, **kwargs)
        finally:
            execution_time = (
                time.perf_counter() - start_time
            )
            emissions_value = tracker.stop()

        final_data = getattr(
            tracker,
            "final_emissions_data",
            None,
        )

        cpu_energy = float(
            getattr(final_data, "cpu_energy", 0)
            if final_data is not None
            else 0
        )
        gpu_energy = float(
            getattr(final_data, "gpu_energy", 0)
            if final_data is not None
            else 0
        )
        ram_energy = float(
            getattr(final_data, "ram_energy", 0)
            if final_data is not None
            else 0
        )
        total_energy = float(
            getattr(final_data, "energy_consumed", 0)
            if final_data is not None
            else 0
        )

        emissions_value = float(
            emissions_value or 0
        )
        cpu_energy = float(cpu_energy or 0)
        gpu_energy = float(gpu_energy or 0)
        ram_energy = float(ram_energy or 0)
        total_energy = float(total_energy or 0)

        carbon_intensity = (
            emissions_value / total_energy
            if total_energy > 0
            else None
        )

        return {
            "result": result,
            "execution_time_sec": execution_time,
            "cpu_energy_kwh": cpu_energy,
            "gpu_energy_kwh": gpu_energy,
            "ram_energy_kwh": ram_energy,
            "total_energy_kwh": total_energy,
            "total_emissions_kg": emissions_value,
            "carbon_intensity_kgco2_kwh": (
                carbon_intensity
            ),
        }

    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    execution_time = (
        time.perf_counter() - start_time
    )

    return {
        "result": result,
        "execution_time_sec": execution_time,
        "cpu_energy_kwh": 0.0,
        "gpu_energy_kwh": 0.0,
        "ram_energy_kwh": 0.0,
        "total_energy_kwh": 0.0,
        "total_emissions_kg": 0.0,
        "carbon_intensity_kgco2_kwh": None,
    }


# =============================================================================
# LOAD MODEL AND TEST 1,000 SAMPLES
# =============================================================================

def main() -> dict[str, str | None]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found:\n"
            f"{MODEL_PATH}\n\n"
            "Place yolo26n_mnist_cpu.pt in the same "
            "folder as this script."
        )

    print("=" * 80)
    print("LOAD-ONLY YOLO26 NANO MNIST CPU TEST")
    print("=" * 80)
    print(f"Model: {MODEL_PATH.resolve()}")
    print("Training: DISABLED")
    print(f"Test samples: {NUM_TEST_SAMPLES}")
    print(f"Device: {DEVICE}")

    for old_file in [
        TELEMETRY_CSV_PATH,
    ]:
        if old_file.exists():
            old_file.unlink()

    # Load saved model only.
    model = YOLO(str(MODEL_PATH))

    model.model.to("cpu")
    model.model.eval()

    model_statistics = get_model_statistics(
        model
    )

    # Download/load official MNIST test split.
    test_dataset = MNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=None,
    )

    if len(test_dataset) < NUM_TEST_SAMPLES:
        raise ValueError(
            f"MNIST test set contains only "
            f"{len(test_dataset)} samples."
        )

    # Reproducible random sample of 1,000 unique images.
    random_generator = np.random.default_rng(SEED)

    selected_indices = random_generator.choice(
        len(test_dataset),
        size=NUM_TEST_SAMPLES,
        replace=False,
    )

    # Warmup is excluded from measured rows.
    warmup_image, _ = test_dataset[
        int(selected_indices[0])
    ]

    warmup_image = warmup_image.convert("RGB")

    with torch.inference_mode():
        _ = model.predict(
            source=warmup_image,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )

    prediction_rows: list[
        dict[str, Any]
    ] = []

    telemetry_rows: list[
        dict[str, Any]
    ] = []

    true_labels: list[int] = []
    predictions: list[int] = []

    overall_start = time.perf_counter()

    for sample_index, dataset_index in enumerate(
        tqdm(
            selected_indices,
            desc="Testing one-by-one",
            unit="image",
        )
    ):
        image, true_label = test_dataset[
            int(dataset_index)
        ]

        # MNIST is grayscale. Convert to RGB for consistent Ultralytics
        # classification preprocessing while preserving the original digit image.
        image = image.convert("RGB")

        true_label = int(true_label)

        def predict_one() -> Any:
            with torch.inference_mode():
                return model.predict(
                    source=image,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    verbose=False,
                )[0]

        result_info = run_with_energy_tracking(
            predict_one,
            project_name=(
                "yolo26n_mnist_cpu_inference"
            ),
            output_file=CODECARBON_CSV_PATH.name,
        )

        result = result_info["result"]

        if result.probs is None:
            raise RuntimeError(
                "Loaded model is not returning "
                "classification probabilities."
            )

        probabilities = (
            result.probs.data
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        prediction = int(
            np.argmax(probabilities)
        )

        raw_score = float(
            probabilities[prediction]
        )

        confidence, margin, entropy = (
            prediction_quality(probabilities)
        )

        execution_time = float(
            result_info["execution_time_sec"]
        )

        input_tokens = (
            IMAGE_SIZE * IMAGE_SIZE
        )
        output_tokens = N_CLASSES
        total_tokens = (
            input_tokens + output_tokens
        )

        total_energy = float(
            result_info["total_energy_kwh"]
            or 0.0
        )

        joules_total = (
            total_energy * 3_600_000
        )

        energy_per_token_kwh = (
            total_energy / total_tokens
            if total_tokens > 0
            else 0.0
        )

        joules_per_token = (
            joules_total / total_tokens
            if total_tokens > 0
            else 0.0
        )

        watts_estimated = (
            joules_total / execution_time
            if execution_time > 0
            else 0.0
        )

        correct = (
            prediction == true_label
        )

        prediction_rows.append(
            {
                "sample_index": sample_index,
                "mnist_dataset_index": int(
                    dataset_index
                ),
                "true_label": true_label,
                "true_class": CLASS_NAMES[
                    true_label
                ],
                "prediction": prediction,
                "predicted_class": CLASS_NAMES[
                    prediction
                ],
                "correct": correct,
                "raw_score": round(
                    raw_score,
                    8,
                ),
                "confidence_score": confidence,
                "score_margin": margin,
                "entropy": entropy,
                "execution_time_sec": round(
                    execution_time,
                    10,
                ),
            }
        )

        telemetry_rows.append(
            {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "unique_device_id": (
                    DEVICE_UUID
                ),
                "device_short_id": (
                    DEVICE_SHORT
                ),
                "pc_name": socket.gethostname(),
                "collection_mode": (
                    "yolo26n_mnist_cpu_load_only_test"
                ),

                "sample_index": sample_index,
                "mnist_dataset_index": int(
                    dataset_index
                ),
                "true_label": true_label,
                "prediction": prediction,
                "correct": correct,

                "model_type": (
                    "YOLO26N_MNIST_CLASSIFICATION_CPU"
                ),
                "framework": "Ultralytics",
                "ultralytics_version": (
                    ULTRALYTICS_VERSION
                ),
                "torch_version": TORCH_VERSION,
                "torchvision_version": (
                    TORCHVISION_VERSION
                ),
                "model_path": str(
                    MODEL_PATH.resolve()
                ),
                "inference_device": "cpu",

                "n_layers": (
                    model_statistics[
                        "module_count"
                    ]
                ),
                "parameter_count": (
                    model_statistics[
                        "total_parameters"
                    ]
                ),
                "trainable_parameter_count": (
                    model_statistics[
                        "trainable_parameters"
                    ]
                ),
                "convolution_layers": (
                    model_statistics[
                        "convolution_layers"
                    ]
                ),
                "linear_layers": (
                    model_statistics[
                        "linear_layers"
                    ]
                ),
                "normalization_layers": (
                    model_statistics[
                        "normalization_layers"
                    ]
                ),
                "activation_layers": (
                    model_statistics[
                        "activation_layers"
                    ]
                ),
                "pooling_layers": (
                    model_statistics[
                        "pooling_layers"
                    ]
                ),

                "raw_score": round(
                    raw_score,
                    8,
                ),
                "confidence_score": confidence,
                "score_margin": margin,
                "entropy": entropy,

                "execution_time_sec": round(
                    execution_time,
                    10,
                ),

                "cpu_energy_kwh": (
                    result_info[
                        "cpu_energy_kwh"
                    ]
                ),
                "gpu_energy_kwh": (
                    result_info[
                        "gpu_energy_kwh"
                    ]
                ),
                "ram_energy_kwh": (
                    result_info[
                        "ram_energy_kwh"
                    ]
                ),
                "total_energy_kwh": (
                    result_info[
                        "total_energy_kwh"
                    ]
                ),
                "total_emissions_kg": (
                    result_info[
                        "total_emissions_kg"
                    ]
                ),
                "carbon_intensity_kgco2_kwh": (
                    result_info[
                        "carbon_intensity_kgco2_kwh"
                    ]
                ),
                "codecarbon_version": (
                    CODECARBON_VERSION
                ),

                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": (
                    round(
                        total_tokens
                        / execution_time,
                        4,
                    )
                    if execution_time > 0
                    else None
                ),
                "joules_per_token": round(
                    joules_per_token,
                    8,
                ),
                "energy_per_token_kwh": (
                    round(
                        energy_per_token_kwh,
                        12,
                    )
                ),
                "watts_estimated": round(
                    watts_estimated,
                    8,
                ),

                "cpu_model": CPU_MODEL_NAME,
                "cpu_core_count": (
                    CPU_CORE_COUNT
                ),
                "cpu_thread_count": (
                    CPU_THREAD_COUNT
                ),
                "torch_num_threads": (
                    torch.get_num_threads()
                ),
                "cpu_usage_pct": (
                    get_cpu_usage()
                ),
                "cpu_clock_mhz": (
                    get_cpu_freq()
                ),
                "cpu_cores_used": (
                    get_cpu_cores_used()
                ),
                "ram_usage_pct": (
                    get_ram_usage()
                ),
                "memory_footprint_mb": (
                    get_memory_footprint_mb()
                ),
                "system_ram_total_gb": (
                    SYSTEM_RAM_TOTAL_GB
                ),

                "os_full_name": OS_FULL_NAME,
                "os_name": platform.system(),
                "os_architecture": (
                    platform.machine()
                ),
                "python_version": (
                    PYTHON_VERSION
                ),

                "model_accuracy": None,
                "model_precision_weighted": None,
                "model_recall_weighted": None,
                "model_f1_weighted": None,
                "model_precision_macro": None,
                "model_recall_macro": None,
                "model_macro_f1": None,
            }
        )

        true_labels.append(true_label)
        predictions.append(prediction)

        if len(telemetry_rows) >= FLUSH_EVERY:
            append_rows(
                telemetry_rows,
                TELEMETRY_CSV_PATH,
            )
            telemetry_rows = []

    if telemetry_rows:
        append_rows(
            telemetry_rows,
            TELEMETRY_CSV_PATH,
        )

    total_runtime = (
        time.perf_counter() - overall_start
    )

    prediction_dataframe = pd.DataFrame(
        prediction_rows
    )

    # prediction_dataframe.to_csv(
    #     PREDICTIONS_CSV_PATH,
    #     index=False,
    # )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    report_dictionary = classification_report(
        true_labels,
        predictions,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report_dictionary
    ).transpose()

    # report_dataframe.to_csv(
    #     CLASSIFICATION_REPORT_CSV_PATH
    # )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(N_CLASSES)),
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"true_{class_name}"
            for class_name in CLASS_NAMES
        ],
        columns=[
            f"pred_{class_name}"
            for class_name in CLASS_NAMES
        ],
    )

    # matrix_dataframe.to_csv(
    #     CONFUSION_MATRIX_CSV_PATH
    # )

    model_metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(
            report_dictionary[
                "weighted avg"
            ]["precision"]
        ),
        "recall_weighted": float(
            report_dictionary[
                "weighted avg"
            ]["recall"]
        ),
        "f1_weighted": float(
            report_dictionary[
                "weighted avg"
            ]["f1-score"]
        ),
        "precision_macro": float(
            report_dictionary[
                "macro avg"
            ]["precision"]
        ),
        "recall_macro": float(
            report_dictionary[
                "macro avg"
            ]["recall"]
        ),
        "macro_f1": float(
            report_dictionary[
                "macro avg"
            ]["f1-score"]
        ),
    }

    telemetry_dataframe = pd.read_csv(
        TELEMETRY_CSV_PATH
    )

    telemetry_dataframe[
        "model_accuracy"
    ] = model_metrics["accuracy"]

    telemetry_dataframe[
        "model_precision_weighted"
    ] = model_metrics[
        "precision_weighted"
    ]

    telemetry_dataframe[
        "model_recall_weighted"
    ] = model_metrics[
        "recall_weighted"
    ]

    telemetry_dataframe[
        "model_f1_weighted"
    ] = model_metrics[
        "f1_weighted"
    ]

    telemetry_dataframe[
        "model_precision_macro"
    ] = model_metrics[
        "precision_macro"
    ]

    telemetry_dataframe[
        "model_recall_macro"
    ] = model_metrics[
        "recall_macro"
    ]

    telemetry_dataframe[
        "model_macro_f1"
    ] = model_metrics["macro_f1"]

    telemetry_dataframe.to_csv(
        TELEMETRY_CSV_PATH,
        index=False,
    )

    results = {
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "model_path": str(
            MODEL_PATH.resolve()
        ),
        "training_performed": False,
        "model_loaded_only": True,
        "inference_device": "cpu",
        "dataset": "MNIST test split",
        "total_tested_samples": (
            NUM_TEST_SAMPLES
        ),
        "sampling_method": (
            "1,000 unique random MNIST samples"
        ),
        "random_seed": SEED,
        "accuracy": model_metrics[
            "accuracy"
        ],
        "macro_precision": (
            model_metrics[
                "precision_macro"
            ]
        ),
        "macro_recall": (
            model_metrics["recall_macro"]
        ),
        "macro_f1": (
            model_metrics["macro_f1"]
        ),
        "weighted_precision": (
            model_metrics[
                "precision_weighted"
            ]
        ),
        "weighted_recall": (
            model_metrics[
                "recall_weighted"
            ]
        ),
        "weighted_f1": (
            model_metrics[
                "f1_weighted"
            ]
        ),
        "total_runtime_seconds": float(
            total_runtime
        ),
        "average_inference_seconds": float(
            prediction_dataframe[
                "execution_time_sec"
            ].mean()
        ),
        "median_inference_seconds": float(
            prediction_dataframe[
                "execution_time_sec"
            ].median()
        ),
        "model_statistics": (
            model_statistics
        ),
        "classification_report": (
            report_dictionary
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "device_uuid": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "cpu_model": CPU_MODEL_NAME,
        "cpu_core_count": (
            CPU_CORE_COUNT
        ),
        "cpu_thread_count": (
            CPU_THREAD_COUNT
        ),
        "system_ram_total_gb": (
            SYSTEM_RAM_TOTAL_GB
        ),
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,
        "torchvision_version": (
            TORCHVISION_VERSION
        ),
        "ultralytics_version": (
            ULTRALYTICS_VERSION
        ),
        "codecarbon_available": (
            CODECARBON_AVAILABLE
        ),
        "codecarbon_enabled": (
            ENABLE_CODECARBON
        ),
        "codecarbon_version": (
            CODECARBON_VERSION
        ),
    }

    # RESULTS_JSON_PATH.write_text(
    #     json.dumps(
    #         results,
    #         indent=4,
    #     ),
    #     encoding="utf-8",
    # )

    return_files: dict[
        str,
        str | None,
    ] = {
        "model_pt": str(
            MODEL_PATH.resolve()
        ),
        # "test_predictions_csv": str(
        #     PREDICTIONS_CSV_PATH.resolve()
        # ),
        "test_telemetry_csv": str(
            TELEMETRY_CSV_PATH.resolve()
        ),
        # "test_results_json": str(
        #     RESULTS_JSON_PATH.resolve()
        # ),
        # "classification_report_csv": str(
        #     CLASSIFICATION_REPORT_CSV_PATH.resolve()
        # ),
        # "confusion_matrix_csv": str(
        #     CONFUSION_MATRIX_CSV_PATH.resolve()
        # ),
        # "codecarbon_csv": (
        #     str(
        #         CODECARBON_CSV_PATH.resolve()
        #     )
        #     if CODECARBON_CSV_PATH.exists()
        #     else None
        # ),
        # "output_directory": str(
        #     OUTPUT_ROOT.resolve()
        # ),
        # "return_files_json": str(
        #     RETURN_FILES_JSON_PATH.resolve()
        # ),
    }

    # RETURN_FILES_JSON_PATH.write_text(
    #     json.dumps(
    #         return_files,
    #         indent=4,
    #     ),
    #     encoding="utf-8",
    # )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Macro F1: "
        f"{model_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{model_metrics['f1_weighted']:.4f}"
    )
    print(
        "Average inference time: "
        f"{results['average_inference_seconds']:.6f} sec"
    )

    print("\nRETURN FILES")

    for file_name, file_path in (
        return_files.items()
    ):
        print(
            f"{file_name}: {file_path}"
        )

    return return_files


if __name__ == "__main__":
    RETURN_FILES = main()
