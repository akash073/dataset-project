"""
Load Only: YOLO26 Nano MNIST CPU Inference with Exact Telemetry Columns
=======================================================================

This script DOES NOT train.

It:
1. Loads yolo26n_mnist_cpu.pt
2. Downloads the MNIST test split
3. Selects NUM_TEST_SAMPLES unique test images
4. Runs inference one image at a time on CPU
5. Collects exactly the requested telemetry columns
6. Backfills final accuracy / weighted precision / recall / F1
7. Saves one telemetry CSV

Install:
    pip install -U ultralytics torch torchvision pandas numpy psutil \
        py-cpuinfo codecarbon scikit-learn tqdm pillow pynvml
"""

from __future__ import annotations

import hashlib
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_TEST_SAMPLES = int(
    os.getenv(
        "NUM_TEST_SAMPLES",
        10,
    )
)

SEED = 42

MODEL_PATH = (
    Path.cwd()
    / "yolo26n_mnist_cpu.pt"
)

DATA_ROOT = Path(
    "./mnist_data"
)

OUTPUT_ROOT = (
    Path.cwd()
    / "test_results"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 28

# Inference is intentionally CPU-only.
DEVICE = torch.device(
    "cpu"
)

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

N_CLASSES = len(
    CLASS_NAMES
)


# =============================================================================
# OPTIONAL CODECARBON
# =============================================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = (
        codecarbon.__version__
    )

except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = (
        "unavailable"
    )


# =============================================================================
# OPTIONAL NVML / GPU TELEMETRY
# =============================================================================

try:
    import pynvml

    pynvml.nvmlInit()

    NVML_AVAILABLE = True

    NVML_HANDLE = (
        pynvml
        .nvmlDeviceGetHandleByIndex(0)
    )

except Exception:
    pynvml = None
    NVML_AVAILABLE = False
    NVML_HANDLE = None


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

def get_cpu_model() -> str:
    try:
        import cpuinfo

        return (
            cpuinfo
            .get_cpu_info()
            .get(
                "brand_raw",
                "Unknown",
            )
        )

    except Exception:
        return (
            platform.processor()
            or "Unknown"
        )


def get_os_full_name() -> str:

    system = (
        platform.system()
    )

    architecture = (
        platform.machine()
    )

    if system == "Windows":
        return (
            f"Windows "
            f"{platform.release()} "
            f"{platform.version()} "
            f"{architecture}"
        )

    if system == "Linux":

        try:
            os_information = {}

            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8",
            ) as file_handle:

                for line in file_handle:

                    if "=" in line:
                        key, value = (
                            line
                            .strip()
                            .split(
                                "=",
                                1,
                            )
                        )

                        os_information[
                            key
                        ] = (
                            value
                            .strip('"')
                        )

            return (
                f"{os_information.get('PRETTY_NAME', 'Linux')} "
                f"{architecture}"
            )

        except Exception:
            return (
                f"Linux "
                f"{platform.release()} "
                f"{architecture}"
            )

    if system == "Darwin":
        return (
            f"macOS "
            f"{platform.mac_ver()[0]} "
            f"{architecture}"
        )

    return (
        f"{system} "
        f"{platform.release()} "
        f"{architecture}"
    )


CPU_MODEL_NAME = (
    get_cpu_model()
)

CPU_ARCH = (
    platform.machine()
)

CPU_CORE_COUNT = (
    psutil.cpu_count(
        logical=False
    )
)

CPU_THREAD_COUNT = (
    psutil.cpu_count(
        logical=True
    )
)

# Cross-platform TDP is not reliably available from psutil.
CPU_TDP_W = None

SYSTEM_RAM_TOTAL_GB = round(
    psutil
    .virtual_memory()
    .total
    / (1024 ** 3),
    2,
)

OS_NAME = (
    platform.system()
)

OS_VERSION = (
    platform.version()
)

OS_ARCHITECTURE = (
    platform.machine()
)

OS_FULL_NAME = (
    get_os_full_name()
)

PYTHON_VERSION = (
    sys.version.split()[0]
)

TORCH_VERSION = (
    torch.__version__
)


def make_stable_device_id() -> str:

    raw_value = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return (
        hashlib
        .sha256(
            raw_value
            .encode("utf-8")
        )
        .hexdigest()
    )


DEVICE_UUID = (
    make_stable_device_id()
)

DEVICE_SHORT = (
    DEVICE_UUID[:8]
)

DEVICE_LOG_DIR = (
    OUTPUT_ROOT
    / DEVICE_SHORT
)

DEVICE_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TELEMETRY_CSV_PATH = (
    DEVICE_LOG_DIR
    / "yolo_test_dataset.csv"
)

CODECARBON_DIR = (
    OUTPUT_ROOT
    / "codecarbon"
)

CODECARBON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CODECARBON_FILE_NAME = (
    "codecarbon_yolo_mnist.csv"
)


# =============================================================================
# EXACT OUTPUT COLUMN ORDER
# =============================================================================

TELEMETRY_COLUMNS = [
    # Identity
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "collection_mode",

    # Sample
    "sample_index",
    "true_label",
    "prediction",
    "correct",

    # Model identity
    "model_type",
    "parameters",
    "model_flops",

    # Prediction quality
    "confidence_score",
    "logit_margin",
    "entropy",

    # Timing
    "execution_time_sec",

    # CodeCarbon energy
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "total_energy_kwh",
    "total_emissions_kg",
    "carbon_intensity_kgco2_kwh",
    "codecarbon_version",

    # Efficiency derived
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens_per_second",
    "joules_per_token",
    "energy_per_token_kwh",
    "watts_estimated",
    "gpu_energy_pct_of_total",
    "cpu_energy_pct_of_total",

    # CPU hardware
    "cpu_model",
    "cpu_architecture",
    "cpu_core_count",
    "cpu_thread_count",
    "cpu_core",
    "cpu_thread",
    "cpu_tdp_w",
    "cpu_usage_pct",
    "cpu_clock_mhz",
    "cpu_temp_c",
    "cpu_power_draw_w",
    "cpu_cores_used",

    # GPU hardware
    "gpu_model",
    "gpu_core",
    "gpu_thread",
    "gpu_driver_version",
    "gpu_compute_capability",
    "gpu_power_limit_w",
    "gpu_memory_total_mb",
    "gpu_power_draw_w",
    "gpu_utilization_pct",
    "gpu_temp_c",
    "gpu_memory_used_mb",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "cuda_driver_version",
    "cuda_available",
    "device_type",

    # RAM / memory
    "ram_usage_pct",
    "memory_footprint_mb",
    "system_ram_total_gb",

    # Environment
    "os_name",
    "os_version",
    "os_architecture",
    "os_full_name",
    "python_version",
    "torch_version",

    # Final model metrics
    "model_accuracy",
    "model_precision_weighted",
    "model_recall_weighted",
    "model_f1_weighted",
]


# =============================================================================
# CPU HELPERS
# =============================================================================

def get_hostname() -> str:
    return (
        socket.gethostname()
    )


def get_memory_footprint_mb():
    try:
        process = (
            psutil.Process(
                os.getpid()
            )
        )

        return round(
            process
            .memory_info()
            .rss
            / (1024 ** 2),
            4,
        )

    except Exception:
        return None


def get_cpu_usage():
    try:
        return (
            psutil.cpu_percent(
                interval=None
            )
        )

    except Exception:
        return None


def get_cpu_freq():
    try:
        frequency = (
            psutil.cpu_freq()
        )

        if frequency is None:
            return None

        return round(
            frequency.current,
            2,
        )

    except Exception:
        return None


def get_cpu_temp():
    try:
        sensors_function = getattr(
            psutil,
            "sensors_temperatures",
            None,
        )

        if sensors_function is None:
            return None

        temperatures = (
            sensors_function()
        )

        if not temperatures:
            return None

        for key in (
            "coretemp",
            "k10temp",
            "cpu_thermal",
            "acpitz",
        ):

            if key not in temperatures:
                continue

            values = [
                item.current
                for item
                in temperatures[key]
                if (
                    item.current
                    is not None
                    and item.current > 0
                )
            ]

            if values:
                return round(
                    sum(values)
                    / len(values),
                    2,
                )

    except Exception:
        pass

    return None


def get_cpu_power_draw_w():
    # True CPU package power needs vendor/OS-specific sensors.
    return None


def get_cpu_cores_used():
    try:
        return sum(
            1
            for value
            in psutil.cpu_percent(
                percpu=True
            )
            if value > 1.0
        )

    except Exception:
        return None


# =============================================================================
# GPU HELPERS
# =============================================================================

def get_cuda_driver_version():

    if (
        not NVML_AVAILABLE
        or pynvml is None
    ):
        return None

    try:
        value = (
            pynvml
            .nvmlSystemGetDriverVersion()
        )

        if isinstance(
            value,
            bytes,
        ):
            return (
                value.decode(
                    "utf-8"
                )
            )

        return str(
            value
        )

    except Exception:
        return None


CUDA_DRIVER_VERSION = (
    get_cuda_driver_version()
)


def get_gpu_name():

    # We report installed GPU hardware even though
    # inference is intentionally CPU-only.
    if (
        NVML_AVAILABLE
        and NVML_HANDLE is not None
        and pynvml is not None
    ):

        try:
            value = (
                pynvml
                .nvmlDeviceGetName(
                    NVML_HANDLE
                )
            )

            if isinstance(
                value,
                bytes,
            ):
                value = (
                    value.decode(
                        "utf-8"
                    )
                )

            return str(
                value
            )

        except Exception:
            pass

    if torch.cuda.is_available():

        try:
            return (
                torch.cuda
                .get_device_name(0)
            )

        except Exception:
            pass

    return "No GPU"


def get_gpu_core_thread():

    if not torch.cuda.is_available():
        return (
            None,
            None,
        )

    try:
        properties = (
            torch.cuda
            .get_device_properties(0)
        )

        sm_count = (
            properties
            .multi_processor_count
        )

        cores_per_sm = {
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(
            properties.major,
            64,
        )

        gpu_core_count = (
            sm_count
            * cores_per_sm
        )

        gpu_thread_count = (
            sm_count
            * properties
            .max_threads_per_multi_processor
        )

        return (
            gpu_core_count,
            gpu_thread_count,
        )

    except Exception:
        return (
            None,
            None,
        )


GPU_CORE_COUNT, GPU_THREAD_COUNT = (
    get_gpu_core_thread()
)


def get_gpu_static():

    result = {
        "gpu_driver_version":
            CUDA_DRIVER_VERSION,

        "gpu_compute_capability":
            None,

        "gpu_power_limit_w":
            None,

        "gpu_memory_total_mb":
            None,
    }

    if torch.cuda.is_available():

        try:
            properties = (
                torch.cuda
                .get_device_properties(0)
            )

            result[
                "gpu_compute_capability"
            ] = (
                f"{properties.major}."
                f"{properties.minor}"
            )

        except Exception:
            pass

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
        or pynvml is None
    ):
        return result

    try:
        power_limit_mw = (
            pynvml
            .nvmlDeviceGetPowerManagementLimit(
                NVML_HANDLE
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        result[
            "gpu_power_limit_w"
        ] = round(
            power_limit_mw
            / 1000.0,
            2,
        )

        result[
            "gpu_memory_total_mb"
        ] = round(
            memory.total
            / (1024 ** 2),
            2,
        )

    except Exception:
        pass

    return result


GPU_STATIC = (
    get_gpu_static()
)


def get_gpu_metrics():

    result = {
        "gpu_power_draw_w":
            None,

        "gpu_utilization_pct":
            None,

        "gpu_temp_c":
            None,

        "gpu_memory_used_mb":
            None,

        "gpu_sm_clock_mhz":
            None,

        "gpu_memory_clock_mhz":
            None,
    }

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
        or pynvml is None
    ):
        return result

    try:
        power_mw = (
            pynvml
            .nvmlDeviceGetPowerUsage(
                NVML_HANDLE
            )
        )

        utilization = (
            pynvml
            .nvmlDeviceGetUtilizationRates(
                NVML_HANDLE
            )
        )

        temperature = (
            pynvml
            .nvmlDeviceGetTemperature(
                NVML_HANDLE,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        sm_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_SM,
            )
        )

        memory_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        result.update(
            {
                "gpu_power_draw_w":
                    round(
                        power_mw
                        / 1000.0,
                        2,
                    ),

                "gpu_utilization_pct":
                    utilization.gpu,

                "gpu_temp_c":
                    temperature,

                "gpu_memory_used_mb":
                    round(
                        memory.used
                        / (1024 ** 2),
                        2,
                    ),

                "gpu_sm_clock_mhz":
                    sm_clock,

                "gpu_memory_clock_mhz":
                    memory_clock,
            }
        )

    except Exception:
        pass

    return result


# =============================================================================
# MODEL STATISTICS
# =============================================================================

def get_model_statistics(
    model: YOLO,
) -> dict[str, int]:

    torch_model = (
        model.model
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in torch_model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in torch_model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters":
            int(
                total_parameters
            ),

        "trainable_parameters":
            int(
                trainable_parameters
            ),
    }


def get_model_flops(
    model: YOLO,
):

    """
    Return approximate FLOPs for one IMAGE_SIZE input.

    Ultralytics get_flops() reports GFLOPs, so convert
    it to FLOPs for the requested model_flops column.
    """

    try:
        from ultralytics.utils.torch_utils import (
            get_flops,
        )

        gflops = get_flops(
            model.model,
            imgsz=IMAGE_SIZE,
        )

        if gflops is None:
            return None

        return int(
            float(gflops)
            * 1_000_000_000
        )

    except Exception:
        return None


# =============================================================================
# PREDICTION QUALITY
# =============================================================================

def prediction_quality(
    probabilities: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:

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
        probabilities
        / probabilities.sum()
    )

    sorted_probabilities = (
        np.sort(
            probabilities
        )[::-1]
    )

    confidence_score = float(
        sorted_probabilities[0]
    )

    if (
        len(
            sorted_probabilities
        )
        > 1
    ):
        # For softmax probabilities:
        # log(p1) - log(p2) equals the difference
        # between the corresponding logits.
        logit_margin = float(
            np.log(
                sorted_probabilities[0]
            )
            - np.log(
                sorted_probabilities[1]
            )
        )

    else:
        logit_margin = 0.0

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )

    return (
        round(
            confidence_score,
            6,
        ),
        round(
            logit_margin,
            6,
        ),
        round(
            entropy,
            6,
        ),
    )


# =============================================================================
# CSV HELPER
# =============================================================================

def append_rows(
    rows: list[
        dict[str, Any]
    ],
    output_path: Path,
) -> None:

    if not rows:
        return

    new_dataframe = (
        pd.DataFrame(
            rows
        )
        .reindex(
            columns=TELEMETRY_COLUMNS
        )
    )

    if output_path.exists():

        old_dataframe = (
            pd.read_csv(
                output_path,
                on_bad_lines="skip",
            )
            .reindex(
                columns=TELEMETRY_COLUMNS
            )
        )

        final_dataframe = (
            pd.concat(
                [
                    old_dataframe,
                    new_dataframe,
                ],
                ignore_index=True,
            )
        )

    else:
        final_dataframe = (
            new_dataframe
        )

    final_dataframe.to_csv(
        output_path,
        index=False,
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

        tracker = (
            EmissionsTracker(
                project_name=
                    project_name,

                output_dir=str(
                    CODECARBON_DIR
                ),

                output_file=
                    output_file,

                log_level=
                    "error",

                save_to_file=
                    True,
            )
        )

        tracker.start()

        start_time = (
            time.perf_counter()
        )

        try:
            result = (
                function(
                    *args,
                    **kwargs,
                )
            )

        finally:
            execution_time = (
                time.perf_counter()
                - start_time
            )

            emissions_value = (
                tracker.stop()
            )

        final_data = getattr(
            tracker,
            "final_emissions_data",
            None,
        )

        cpu_energy = float(
            getattr(
                final_data,
                "cpu_energy",
                0,
            )
            if final_data is not None
            else 0
        )

        gpu_energy = float(
            getattr(
                final_data,
                "gpu_energy",
                0,
            )
            if final_data is not None
            else 0
        )

        ram_energy = float(
            getattr(
                final_data,
                "ram_energy",
                0,
            )
            if final_data is not None
            else 0
        )

        total_energy = float(
            getattr(
                final_data,
                "energy_consumed",
                0,
            )
            if final_data is not None
            else 0
        )

        emissions_value = float(
            emissions_value
            or 0
        )

        cpu_energy = float(
            cpu_energy
            or 0
        )

        gpu_energy = float(
            gpu_energy
            or 0
        )

        ram_energy = float(
            ram_energy
            or 0
        )

        total_energy = float(
            total_energy
            or 0
        )

        carbon_intensity = (
            emissions_value
            / total_energy
            if total_energy > 0
            else None
        )

        return {
            "result":
                result,

            "execution_time_sec":
                execution_time,

            "cpu_energy_kwh":
                cpu_energy,

            "gpu_energy_kwh":
                gpu_energy,

            "ram_energy_kwh":
                ram_energy,

            "total_energy_kwh":
                total_energy,

            "total_emissions_kg":
                emissions_value,

            "carbon_intensity_kgco2_kwh":
                carbon_intensity,
        }

    start_time = (
        time.perf_counter()
    )

    result = function(
        *args,
        **kwargs,
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    return {
        "result":
            result,

        "execution_time_sec":
            execution_time,

        "cpu_energy_kwh":
            0.0,

        "gpu_energy_kwh":
            0.0,

        "ram_energy_kwh":
            0.0,

        "total_energy_kwh":
            0.0,

        "total_emissions_kg":
            0.0,

        "carbon_intensity_kgco2_kwh":
            None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model file not found:\n"
            f"{MODEL_PATH}\n\n"
            "Place yolo26n_mnist_cpu.pt "
            "in the same folder as this script."
        )

    print(
        "=" * 80
    )

    print(
        "LOAD-ONLY YOLO26 NANO "
        "MNIST CPU TEST"
    )

    print(
        "=" * 80
    )

    print(
        "Model:",
        MODEL_PATH.resolve(),
    )

    print(
        "Training: DISABLED"
    )

    print(
        "Test samples:",
        NUM_TEST_SAMPLES,
    )

    print(
        "Device:",
        DEVICE,
    )


    if TELEMETRY_CSV_PATH.exists():

        TELEMETRY_CSV_PATH.unlink()


    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    model.model.to(
        DEVICE
    )

    model.model.eval()


    model_statistics = (
        get_model_statistics(
            model
        )
    )

    parameters = (
        model_statistics[
            "total_parameters"
        ]
    )

    model_flops = (
        get_model_flops(
            model
        )
    )

    model_name = (
        "YOLO26N"
    )


    print(
        "Parameters:",
        parameters,
    )

    print(
        "Model FLOPs:",
        model_flops,
    )


    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    test_dataset = MNIST(
        root=str(
            DATA_ROOT
        ),
        train=False,
        download=True,
        transform=None,
    )


    if (
        len(
            test_dataset
        )
        < NUM_TEST_SAMPLES
    ):

        raise ValueError(
            "MNIST test set contains only "
            f"{len(test_dataset)} samples."
        )


    random_generator = (
        np.random.default_rng(
            SEED
        )
    )


    selected_indices = (
        random_generator.choice(
            len(
                test_dataset
            ),
            size=
                NUM_TEST_SAMPLES,
            replace=
                False,
        )
    )


    # -------------------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------------------

    warmup_image, _ = (
        test_dataset[
            int(
                selected_indices[0]
            )
        ]
    )

    warmup_image = (
        warmup_image
        .convert(
            "RGB"
        )
    )


    with torch.inference_mode():

        _ = model.predict(
            source=
                warmup_image,

            imgsz=
                IMAGE_SIZE,

            device=
                "cpu",

            verbose=
                False,
        )


    # -------------------------------------------------------------------------
    # Test loop
    # -------------------------------------------------------------------------

    telemetry_rows = []

    true_labels = []

    predictions = []


    for (
        sample_index,
        dataset_index,
    ) in enumerate(
        tqdm(
            selected_indices,
            desc=
                "Testing one-by-one",
            unit=
                "image",
        )
    ):

        image, true_label = (
            test_dataset[
                int(
                    dataset_index
                )
            ]
        )


        image = (
            image
            .convert(
                "RGB"
            )
        )


        true_label = int(
            true_label
        )


        def predict_one():

            with torch.inference_mode():

                return (
                    model.predict(
                        source=
                            image,

                        imgsz=
                            IMAGE_SIZE,

                        device=
                            "cpu",

                        verbose=
                            False,
                    )[0]
                )


        result_info = (
            run_with_energy_tracking(
                predict_one,

                project_name=
                    "yolo26n_mnist_cpu_inference",

                output_file=
                    CODECARBON_FILE_NAME,
            )
        )


        result = (
            result_info[
                "result"
            ]
        )


        if result.probs is None:

            raise RuntimeError(
                "Loaded model is not "
                "returning classification "
                "probabilities."
            )


        probabilities = (
            result.probs.data
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        prediction = int(
            np.argmax(
                probabilities
            )
        )


        (
            confidence_score,
            logit_margin,
            entropy,
        ) = prediction_quality(
            probabilities
        )


        exec_time = float(
            result_info[
                "execution_time_sec"
            ]
        )


        # Native MNIST pixel/value proxy.
        input_tokens = (
            IMAGE_SIZE
            * IMAGE_SIZE
        )

        # Classification output space.
        output_tokens = (
            N_CLASSES
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )


        cpu_energy = float(
            result_info[
                "cpu_energy_kwh"
            ]
            or 0.0
        )

        gpu_energy = float(
            result_info[
                "gpu_energy_kwh"
            ]
            or 0.0
        )

        ram_energy = float(
            result_info[
                "ram_energy_kwh"
            ]
            or 0.0
        )

        total_energy = float(
            result_info[
                "total_energy_kwh"
            ]
            or 0.0
        )

        emissions_value = float(
            result_info[
                "total_emissions_kg"
            ]
            or 0.0
        )

        carbon_intensity = (
            result_info[
                "carbon_intensity_kgco2_kwh"
            ]
        )


        joules_total = (
            total_energy
            * 3_600_000
        )


        energy_per_token_kwh = (
            total_energy
            / total_tokens
            if total_tokens > 0
            else 0.0
        )


        joules_per_token = (
            joules_total
            / total_tokens
            if total_tokens > 0
            else 0.0
        )


        watts_estimated = (
            joules_total
            / exec_time
            if exec_time > 0
            else 0.0
        )


        gpu_energy_pct = (
            gpu_energy
            / total_energy
            * 100
            if total_energy > 0
            else 0.0
        )


        cpu_energy_pct = (
            cpu_energy
            / total_energy
            * 100
            if total_energy > 0
            else 0.0
        )


        correct = (
            prediction
            == true_label
        )


        gpu_metrics = (
            get_gpu_metrics()
        )


        row = {
            # --- Identity ---
            "timestamp":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "unique_device_id":
                DEVICE_UUID,

            "device_short_id":
                DEVICE_SHORT,

            "pc_name":
                get_hostname(),

            "collection_mode":
                "automated_edge",

            # --- Sample ---
            "sample_index":
                sample_index,

            "true_label":
                int(
                    true_label
                ),

            "prediction":
                int(
                    prediction
                ),

            "correct":
                correct,

            # --- Model identity ---
            "model_type":
                model_name,

            "parameters":
                parameters,

            "model_flops":
                model_flops,

            # --- Prediction quality ---
            "confidence_score":
                confidence_score,

            "logit_margin":
                logit_margin,

            "entropy":
                entropy,

            # --- Timing ---
            "execution_time_sec":
                round(
                    exec_time,
                    10,
                ),

            # --- CodeCarbon energy ---
            "cpu_energy_kwh":
                cpu_energy,

            "gpu_energy_kwh":
                gpu_energy,

            "ram_energy_kwh":
                ram_energy,

            "total_energy_kwh":
                total_energy,

            "total_emissions_kg":
                emissions_value,

            "carbon_intensity_kgco2_kwh":
                carbon_intensity,

            "codecarbon_version":
                CODECARBON_VERSION,

            # --- Efficiency derived ---
            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens,

            "tokens_per_second":
                (
                    round(
                        total_tokens
                        / exec_time,
                        4,
                    )
                    if exec_time > 0
                    else None
                ),

            "joules_per_token":
                round(
                    joules_per_token,
                    8,
                ),

            "energy_per_token_kwh":
                round(
                    energy_per_token_kwh,
                    12,
                ),

            "watts_estimated":
                round(
                    watts_estimated,
                    8,
                ),

            "gpu_energy_pct_of_total":
                round(
                    gpu_energy_pct,
                    4,
                ),

            "cpu_energy_pct_of_total":
                round(
                    cpu_energy_pct,
                    4,
                ),

            # --- CPU hardware ---
            "cpu_model":
                CPU_MODEL_NAME,

            "cpu_architecture":
                CPU_ARCH,

            "cpu_core_count":
                CPU_CORE_COUNT,

            "cpu_thread_count":
                CPU_THREAD_COUNT,

            "cpu_core":
                CPU_CORE_COUNT,

            "cpu_thread":
                CPU_THREAD_COUNT,

            "cpu_tdp_w":
                CPU_TDP_W,

            "cpu_usage_pct":
                get_cpu_usage(),

            "cpu_clock_mhz":
                get_cpu_freq(),

            "cpu_temp_c":
                get_cpu_temp(),

            "cpu_power_draw_w":
                get_cpu_power_draw_w(),

            "cpu_cores_used":
                get_cpu_cores_used(),

            # --- GPU hardware ---
            "gpu_model":
                get_gpu_name(),

            "gpu_core":
                GPU_CORE_COUNT,

            "gpu_thread":
                GPU_THREAD_COUNT,

            "gpu_driver_version":
                GPU_STATIC[
                    "gpu_driver_version"
                ],

            "gpu_compute_capability":
                GPU_STATIC[
                    "gpu_compute_capability"
                ],

            "gpu_power_limit_w":
                GPU_STATIC[
                    "gpu_power_limit_w"
                ],

            "gpu_memory_total_mb":
                GPU_STATIC[
                    "gpu_memory_total_mb"
                ],

            "gpu_power_draw_w":
                gpu_metrics.get(
                    "gpu_power_draw_w"
                ),

            "gpu_utilization_pct":
                gpu_metrics.get(
                    "gpu_utilization_pct"
                ),

            "gpu_temp_c":
                gpu_metrics.get(
                    "gpu_temp_c"
                ),

            "gpu_memory_used_mb":
                gpu_metrics.get(
                    "gpu_memory_used_mb"
                ),

            "gpu_sm_clock_mhz":
                gpu_metrics.get(
                    "gpu_sm_clock_mhz"
                ),

            "gpu_memory_clock_mhz":
                gpu_metrics.get(
                    "gpu_memory_clock_mhz"
                ),

            "cuda_driver_version":
                CUDA_DRIVER_VERSION,

            "cuda_available":
                torch.cuda.is_available(),

            "device_type":
                str(
                    DEVICE
                ),

            # --- RAM / memory ---
            "ram_usage_pct":
                psutil
                .virtual_memory()
                .percent,

            "memory_footprint_mb":
                get_memory_footprint_mb(),

            "system_ram_total_gb":
                SYSTEM_RAM_TOTAL_GB,

            # --- Environment ---
            "os_name":
                OS_NAME,

            "os_version":
                OS_VERSION,

            "os_architecture":
                OS_ARCHITECTURE,

            "os_full_name":
                OS_FULL_NAME,

            "python_version":
                PYTHON_VERSION,

            "torch_version":
                TORCH_VERSION,

            # --- Final model metrics ---
            "model_accuracy":
                None,

            "model_precision_weighted":
                None,

            "model_recall_weighted":
                None,

            "model_f1_weighted":
                None,
        }


        telemetry_rows.append(
            row
        )


        true_labels.append(
            true_label
        )

        predictions.append(
            prediction
        )


        if (
            len(
                telemetry_rows
            )
            >= FLUSH_EVERY
        ):

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


    # -------------------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------------------

    accuracy = (
        accuracy_score(
            true_labels,
            predictions,
        )
    )


    report_dictionary = (
        classification_report(
            true_labels,
            predictions,

            labels=
                list(
                    range(
                        N_CLASSES
                    )
                ),

            target_names=
                CLASS_NAMES,

            output_dict=
                True,

            zero_division=
                0,
        )
    )


    model_metrics = {
        "accuracy":
            float(
                accuracy
            ),

        "precision_weighted":
            float(
                report_dictionary[
                    "weighted avg"
                ][
                    "precision"
                ]
            ),

        "recall_weighted":
            float(
                report_dictionary[
                    "weighted avg"
                ][
                    "recall"
                ]
            ),

        "f1_weighted":
            float(
                report_dictionary[
                    "weighted avg"
                ][
                    "f1-score"
                ]
            ),
    }


    # -------------------------------------------------------------------------
    # Backfill final metrics
    # -------------------------------------------------------------------------

    telemetry_dataframe = (
        pd.read_csv(
            TELEMETRY_CSV_PATH
        )
        .reindex(
            columns=
                TELEMETRY_COLUMNS
        )
    )


    telemetry_dataframe[
        "model_accuracy"
    ] = (
        model_metrics[
            "accuracy"
        ]
    )


    telemetry_dataframe[
        "model_precision_weighted"
    ] = (
        model_metrics[
            "precision_weighted"
        ]
    )


    telemetry_dataframe[
        "model_recall_weighted"
    ] = (
        model_metrics[
            "recall_weighted"
        ]
    )


    telemetry_dataframe[
        "model_f1_weighted"
    ] = (
        model_metrics[
            "f1_weighted"
        ]
    )
    # "model_under_attack":              0,   
    telemetry_dataframe[
        "model_under_attack"
    ] = 0

    telemetry_dataframe.to_csv(
        TELEMETRY_CSV_PATH,
        index=False,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 80
    )


    print(
        f"Accuracy: "
        f"{model_metrics['accuracy']:.4f}"
    )


    print(
        f"Weighted Precision: "
        f"{model_metrics['precision_weighted']:.4f}"
    )


    print(
        f"Weighted Recall: "
        f"{model_metrics['recall_weighted']:.4f}"
    )


    print(
        f"Weighted F1: "
        f"{model_metrics['f1_weighted']:.4f}"
    )


    print(
        "\nTelemetry CSV:"
    )

    print(
        TELEMETRY_CSV_PATH.resolve()
    )


    return {
        "test_telemetry_csv":
            str(
                TELEMETRY_CSV_PATH.resolve()
            )
    }


if __name__ == "__main__":
    RETURN_FILES = main()
