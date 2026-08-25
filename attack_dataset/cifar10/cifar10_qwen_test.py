import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from tqdm import tqdm
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

DATA_ROOT = "../data"

OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "qwen2vl_cifar10_test_metrics.csv"

# None = entire CIFAR-10 test set = 10,000 images
# Start with 10 or 100 first
NUM_SAMPLES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.set_grad_enabled(False)


# ============================================================
# CIFAR-10 class names
# ============================================================

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# Optional CodeCarbon
# ============================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = codecarbon.__version__

except Exception:

    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"

    print("CodeCarbon unavailable.")


# ============================================================
# Optional NVML
# ============================================================

try:
    import pynvml

    pynvml.nvmlInit()

    NVML_AVAILABLE = True

    NVML_HANDLE = (
        pynvml.nvmlDeviceGetHandleByIndex(0)
        if torch.cuda.is_available()
        else None
    )

except Exception:

    NVML_AVAILABLE = False
    NVML_HANDLE = None

    print("NVML unavailable.")


# ============================================================
# Optional CPU info
# ============================================================

try:
    import cpuinfo

    cpu_data = cpuinfo.get_cpu_info()

    CPU_MODEL_NAME = cpu_data.get(
        "brand_raw",
        "Unknown"
    )

    CPU_ARCH = cpu_data.get(
        "arch",
        platform.machine()
    )

except Exception:

    CPU_MODEL_NAME = platform.processor()
    CPU_ARCH = platform.machine()


CPU_TDP_W = None


# ============================================================
# Environment
# ============================================================

PYTHON_VERSION = sys.version.split()[0]
TORCH_VERSION = torch.__version__

OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()

SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total / (1024 ** 3),
    2
)

CPU_CORE_COUNT = psutil.cpu_count(
    logical=False
)

CPU_THREAD_COUNT = psutil.cpu_count(
    logical=True
)


# ============================================================
# OS full name
# ============================================================

def get_os_full_name():

    system = platform.system()

    if system == "Windows":

        return (
            f"Windows {platform.release()} "
            f"{platform.machine()}"
        )

    elif system == "Linux":

        try:

            info = {}

            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    if "=" in line:

                        key, value = line.strip().split(
                            "=",
                            1
                        )

                        info[key] = value.strip('"')

            name = info.get(
                "PRETTY_NAME",
                "Linux"
            )

            return (
                f"{name} "
                f"{platform.machine()}"
            )

        except Exception:

            return (
                f"Linux {platform.release()} "
                f"{platform.machine()}"
            )

    elif system == "Darwin":

        return (
            f"macOS "
            f"{platform.mac_ver()[0]} "
            f"{platform.machine()}"
        )

    return (
        f"{system} "
        f"{platform.release()} "
        f"{platform.machine()}"
    )


OS_FULL_NAME = get_os_full_name()


# ============================================================
# Device identity
# ============================================================

def make_stable_device_id():

    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]


def get_hostname():

    return socket.gethostname()


# ============================================================
# GPU static information
# ============================================================

def get_cuda_driver_version():

    if not NVML_AVAILABLE:
        return None

    try:

        version = pynvml.nvmlSystemGetDriverVersion()

        if isinstance(version, bytes):
            version = version.decode("utf-8")

        return version

    except Exception:

        return None


CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static():

    result = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_memory_total_mb": None,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None
    }

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):
        return result

    try:

        power_limit = (
            pynvml.nvmlDeviceGetPowerManagementLimit(
                NVML_HANDLE
            )
        )

        memory = (
            pynvml.nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        result["gpu_power_limit_w"] = round(
            power_limit / 1000,
            2
        )

        result["gpu_memory_total_mb"] = round(
            memory.total / (1024 ** 2),
            2
        )

    except Exception:
        pass

    if torch.cuda.is_available():

        try:

            props = torch.cuda.get_device_properties(0)

            result["gpu_compute_capability"] = (
                f"{props.major}.{props.minor}"
            )

        except Exception:
            pass

    return result


GPU_STATIC = get_gpu_static()


# ============================================================
# GPU core / thread approximation
# ============================================================

def get_gpu_core_thread():

    if not torch.cuda.is_available():
        return None, None

    try:

        props = torch.cuda.get_device_properties(0)

        sm_count = props.multi_processor_count
        major = props.major

        cores_per_sm = {
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128
        }.get(
            major,
            64
        )

        gpu_core_count = (
            sm_count * cores_per_sm
        )

        gpu_thread_count = (
            sm_count *
            props.max_threads_per_multi_processor
        )

        return (
            gpu_core_count,
            gpu_thread_count
        )

    except Exception:

        return None, None


GPU_CORE_COUNT, GPU_THREAD_COUNT = (
    get_gpu_core_thread()
)


# ============================================================
# Hardware helper functions
# ============================================================

def get_gpu_name():

    if torch.cuda.is_available():

        try:
            return torch.cuda.get_device_name(0)

        except Exception:
            return "Unknown GPU"

    return "No GPU"


def get_cpu_usage():

    return psutil.cpu_percent(
        interval=None
    )


def get_cpu_freq():

    try:

        freq = psutil.cpu_freq()

        if freq:

            return round(
                freq.current,
                2
            )

    except Exception:
        pass

    return None


def get_cpu_temp():

    try:

        temps = psutil.sensors_temperatures()

        if not temps:
            return None

        for name in (
            "coretemp",
            "k10temp",
            "cpu_thermal",
            "acpitz"
        ):

            if name in temps:

                values = [
                    x.current
                    for x in temps[name]
                    if x.current is not None
                ]

                if values:

                    return round(
                        sum(values) / len(values),
                        2
                    )

    except Exception:
        pass

    return None


def get_cpu_power_draw_w():

    return None


def get_cpu_cores_used():

    try:

        usage = psutil.cpu_percent(
            percpu=True
        )

        return sum(
            1
            for value in usage
            if value > 1
        )

    except Exception:

        return None


def get_memory_footprint_mb():

    try:

        process = psutil.Process(
            os.getpid()
        )

        return round(
            process.memory_info().rss /
            (1024 ** 2),
            4
        )

    except Exception:

        return None


# ============================================================
# GPU runtime metrics
# ============================================================

def get_gpu_metrics():

    result = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None
    }

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):

        return result

    try:

        power = pynvml.nvmlDeviceGetPowerUsage(
            NVML_HANDLE
        )

        utilization = (
            pynvml.nvmlDeviceGetUtilizationRates(
                NVML_HANDLE
            )
        )

        temperature = (
            pynvml.nvmlDeviceGetTemperature(
                NVML_HANDLE,
                pynvml.NVML_TEMPERATURE_GPU
            )
        )

        memory = (
            pynvml.nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        sm_clock = (
            pynvml.nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_SM
            )
        )

        memory_clock = (
            pynvml.nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_MEM
            )
        )

        result = {
            "gpu_power_draw_w":
                round(power / 1000, 2),

            "gpu_utilization_pct":
                utilization.gpu,

            "gpu_temp_c":
                temperature,

            "gpu_memory_used_mb":
                round(
                    memory.used /
                    (1024 ** 2),
                    2
                ),

            "gpu_sm_clock_mhz":
                sm_clock,

            "gpu_memory_clock_mhz":
                memory_clock
        }

    except Exception:
        pass

    return result


# ============================================================
# Load CIFAR-10
# ============================================================

print("\nLoading CIFAR-10...")

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=True
)

print(
    "CIFAR-10 test samples:",
    len(test_dataset)
)


# ============================================================
# Load Qwen2-VL
# ============================================================

print(
    "\nLoading:",
    MODEL_ID
)


processor = AutoProcessor.from_pretrained(
    MODEL_ID,

    # CIFAR-10 is 32x32.
    # Use small visual resolution for efficiency.
    min_pixels=56 * 56,
    max_pixels=112 * 112
)


if torch.cuda.is_available():

    model = (
        Qwen2VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    )

else:

    model = (
        Qwen2VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        )
    )

    model = model.to(DEVICE)


model.eval()


# ============================================================
# Parameter count
# ============================================================

MODEL_PARAMETERS = sum(
    p.numel()
    for p in model.parameters()
)

MODEL_NAME = "Qwen2-VL-2B-Instruct-CIFAR10"

MODEL_FLOPS = None


print(
    f"Parameters: "
    f"{MODEL_PARAMETERS:,}"
)


# ============================================================
# Helper: score a complete text class
# ============================================================

def get_class_logprob(
    base_inputs,
    class_name
):

    """
    Calculate log P(class_name | image, prompt).

    This supports labels that may contain more than one token.
    """

    tokenizer = processor.tokenizer

    class_token_ids = tokenizer.encode(
        class_name,
        add_special_tokens=False
    )

    input_ids = base_inputs["input_ids"]

    total_logprob = 0.0

    current_input_ids = input_ids

    # Preserve multimodal inputs
    model_inputs = {
        key: value
        for key, value in base_inputs.items()
    }

    for token_id in class_token_ids:

        model_inputs["input_ids"] = current_input_ids

        if "attention_mask" in model_inputs:

            model_inputs["attention_mask"] = torch.ones_like(
                current_input_ids
            )

        with torch.inference_mode():

            outputs = model(
                **model_inputs,
                return_dict=True
            )

        next_logits = outputs.logits[
            0,
            -1,
            :
        ]

        log_probs = torch.log_softmax(
            next_logits,
            dim=-1
        )

        total_logprob += float(
            log_probs[token_id].item()
        )

        next_token = torch.tensor(
            [[token_id]],
            device=current_input_ids.device,
            dtype=current_input_ids.dtype
        )

        current_input_ids = torch.cat(
            [
                current_input_ids,
                next_token
            ],
            dim=1
        )

    return total_logprob


# ============================================================
# Qwen CIFAR-10 inference
# ============================================================

def run_qwen_inference(image):

    image = image.convert("RGB")

    class_text = ", ".join(
        CIFAR10_CLASSES
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": (
                        "This is a CIFAR-10 image. "
                        "Classify the image as exactly one of "
                        f"these classes: {class_text}. "
                        "Return only the class name."
                    )
                }
            ]
        }
    ]


    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )


    model_device = next(
        model.parameters()
    ).device


    inputs = {
        key:
            value.to(model_device)
            if torch.is_tensor(value)
            else value

        for key, value
        in inputs.items()
    }


    # ========================================================
    # Input token count
    # ========================================================

    if "attention_mask" in inputs:

        input_tokens = int(
            inputs[
                "attention_mask"
            ].sum().item()
        )

    else:

        input_tokens = int(
            inputs[
                "input_ids"
            ].numel()
        )


    # ========================================================
    # Score all 10 CIFAR-10 classes
    # ========================================================

    class_scores = []


    for class_name in CIFAR10_CLASSES:

        score = get_class_logprob(
            inputs,
            class_name
        )

        class_scores.append(
            score
        )


    class_scores_tensor = torch.tensor(
        class_scores,
        dtype=torch.float32
    )


    class_probs = torch.softmax(
        class_scores_tensor,
        dim=0
    )


    prediction = int(
        torch.argmax(
            class_probs
        ).item()
    )


    confidence_score = float(
        class_probs[
            prediction
        ].item()
    )


    # ========================================================
    # Logit / score margin
    # ========================================================

    top2_scores = torch.topk(
        class_scores_tensor,
        k=2
    ).values


    logit_margin = float(
        (
            top2_scores[0]
            -
            top2_scores[1]
        ).item()
    )


    # ========================================================
    # Entropy over 10 classes
    # ========================================================

    entropy = float(
        -(
            class_probs
            *
            torch.log(
                class_probs + 1e-12
            )
        ).sum().item()
    )


    # Generated class may be multiple tokens.
    predicted_class_name = (
        CIFAR10_CLASSES[
            prediction
        ]
    )

    output_tokens = len(
        processor.tokenizer.encode(
            predicted_class_name,
            add_special_tokens=False
        )
    )


    total_tokens = (
        input_tokens
        +
        output_tokens
    )


    return {

        "prediction":
            prediction,

        "predicted_class":
            predicted_class_name,

        "confidence_score":
            confidence_score,

        "logit_margin":
            logit_margin,

        "entropy":
            entropy,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens
    }


# ============================================================
# CodeCarbon wrapper
# ============================================================

def run_with_energy_tracking(
    inference_fn,
    image
):

    output_energy_dir = (
        OUTPUT_DIR / "codecarbon"
    )

    output_energy_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if CODECARBON_AVAILABLE:

        tracker = EmissionsTracker(
            project_name=
                "qwen2vl_cifar10_test",

            output_dir=
                str(output_energy_dir),

            output_file=
                "codecarbon_qwen_cifar10.csv",

            log_level="error",

            save_to_file=True
        )


        tracker.start()


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        start = time.perf_counter()


        result = inference_fn(image)


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        exec_time = (
            time.perf_counter()
            -
            start
        )


        emissions_value = tracker.stop()


        final_data = getattr(
            tracker,
            "final_emissions_data",
            None
        )


        cpu_energy = float(
            getattr(
                final_data,
                "cpu_energy",
                0
            )
            or 0
        )


        gpu_energy = float(
            getattr(
                final_data,
                "gpu_energy",
                0
            )
            or 0
        )


        ram_energy = float(
            getattr(
                final_data,
                "ram_energy",
                0
            )
            or 0
        )


        total_energy = float(
            getattr(
                final_data,
                "energy_consumed",
                0
            )
            or 0
        )


        carbon_intensity = None


        if (
            emissions_value is not None
            and total_energy > 0
        ):

            carbon_intensity = (
                float(emissions_value)
                /
                total_energy
            )


    else:

        if torch.cuda.is_available():
            torch.cuda.synchronize()


        start = time.perf_counter()


        result = inference_fn(image)


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        exec_time = (
            time.perf_counter()
            -
            start
        )


        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None


    gpu_metrics = get_gpu_metrics()


    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        gpu_metrics
    )


# ============================================================
# Build row
# ============================================================

def build_row(
    sample_index,
    true_label,
    result,
    exec_time,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions_value,
    carbon_intensity,
    gpu_metrics
):

    prediction = result[
        "prediction"
    ]


    correct = int(
        prediction
        ==
        true_label
    )


    input_tokens = result[
        "input_tokens"
    ]

    output_tokens = result[
        "output_tokens"
    ]

    total_tokens = result[
        "total_tokens"
    ]


    energy_per_token_kwh = 0.0
    joules_per_token = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0


    if (
        total_energy > 0
        and total_tokens > 0
    ):

        energy_per_token_kwh = (
            total_energy
            /
            total_tokens
        )


        joules_per_token = (
            total_energy
            *
            3_600_000
            /
            total_tokens
        )


        if exec_time > 0:

            watts_estimated = (
                total_energy
                *
                3_600_000
                /
                exec_time
            )


        gpu_energy_pct = (
            gpu_energy
            /
            total_energy
            *
            100
        )


        cpu_energy_pct = (
            cpu_energy
            /
            total_energy
            *
            100
        )


    return {

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
            true_label,

        "true_class":
            CIFAR10_CLASSES[
                true_label
            ],

        "prediction":
            prediction,

        "predicted_class":
            result[
                "predicted_class"
            ],

        "correct":
            correct,


        # --- Model identity ---
        "model_type":
            MODEL_NAME,

        "parameters":
            MODEL_PARAMETERS,

        "model_flops":
            MODEL_FLOPS,


        # --- Prediction quality ---
        "confidence_score":
            round(
                result[
                    "confidence_score"
                ],
                6
            ),

        "logit_margin":
            round(
                result[
                    "logit_margin"
                ],
                6
            ),

        "entropy":
            round(
                result[
                    "entropy"
                ],
                6
            ),


        # --- Timing ---
        "execution_time_sec":
            round(
                exec_time,
                10
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
            round(
                total_tokens /
                exec_time,
                4
            )
            if exec_time > 0
            else None,

        "joules_per_token":
            round(
                joules_per_token,
                8
            ),

        "energy_per_token_kwh":
            round(
                energy_per_token_kwh,
                12
            ),

        "watts_estimated":
            round(
                watts_estimated,
                4
            ),

        "gpu_energy_pct_of_total":
            round(
                gpu_energy_pct,
                2
            ),

        "cpu_energy_pct_of_total":
            round(
                cpu_energy_pct,
                2
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
            str(DEVICE),


        # --- RAM / memory ---
        "ram_usage_pct":
            psutil.virtual_memory().percent,

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


        # --- Final metrics backfilled later ---
        "model_accuracy":
            None,

        "model_precision_weighted":
            None,

        "model_recall_weighted":
            None,

        "model_f1_weighted":
            None
    }


# ============================================================
# Main test
# ============================================================

def main():

    limit = (
        len(test_dataset)
        if NUM_SAMPLES is None
        else min(
            NUM_SAMPLES,
            len(test_dataset)
        )
    )


    print(
        f"\nTesting {MODEL_NAME}"
    )

    print(
        f"Samples: {limit}"
    )

    print(
        f"Device: {DEVICE}"
    )


    rows = []


    for i in tqdm(
        range(limit),
        desc="Qwen2-VL CIFAR-10",
        unit="sample"
    ):

        image, true_label = (
            test_dataset[i]
        )


        (
            result,
            exec_time,
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            emissions_value,
            carbon_intensity,
            gpu_metrics

        ) = run_with_energy_tracking(
            run_qwen_inference,
            image
        )


        row = build_row(
            sample_index=i,
            true_label=int(
                true_label
            ),
            result=result,
            exec_time=exec_time,
            cpu_energy=cpu_energy,
            gpu_energy=gpu_energy,
            ram_energy=ram_energy,
            total_energy=total_energy,
            emissions_value=emissions_value,
            carbon_intensity=
                carbon_intensity,
            gpu_metrics=gpu_metrics
        )


        rows.append(row)


        # Save continuously
        pd.DataFrame(
            rows
        ).to_csv(
            OUTPUT_CSV,
            index=False
        )


    # ========================================================
    # Final metrics
    # ========================================================

    df = pd.DataFrame(rows)


    y_true = df[
        "true_label"
    ].values


    y_pred = df[
        "prediction"
    ].values


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # Backfill final metrics
    # ========================================================

    df[
        "model_accuracy"
    ] = accuracy


    df[
        "model_precision_weighted"
    ] = precision


    df[
        "model_recall_weighted"
    ] = recall


    df[
        "model_f1_weighted"
    ] = f1


    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    print(
        "\n================================="
    )

    print(
        "Qwen2-VL CIFAR-10 TEST RESULTS"
    )

    print(
        "================================="
    )

    print(
        f"Samples   : {limit}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()