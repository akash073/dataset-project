from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
)
from transformers.generation.utils import GenerationMixin


# ============================================================
# Configuration
# ============================================================
# Retrieves env var 'NUM_TEST_SAMPLES', defaults to 10 if not set
NUM_TEST_SAMPLES = int(os.getenv("NUM_TEST_SAMPLES", 10))

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

# Use the instruct model directly from Hugging Face.
MODEL_ID = "fal/moondream2-docci-instruct"

# IMPORTANT:
# Pin the DoCCI instruct model revision so its weights and bundled
# custom architecture stay reproducible and compatible.
MODEL_REVISION = "3ec40c7b6b5d87bc0c51edee45e21f5f29b449d8"

MNIST_ROOT = DATA_ROOT

OUTPUT_CSV = Path(
    DEVICE_LOG_DIR / "moondream2_docci_instruct_mnist_test_predictions.csv"
)

# METRICS_JSON = Path(
#     "./moondream2_docci_instruct_mnist_test_metrics.json"
# )

# 10 for quick testing.
# 1000 for a larger experiment.
# None for all 10,000 MNIST test images.
MAX_TEST_SAMPLES = NUM_TEST_SAMPLES

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

QUESTION = (
    "Classify this MNIST handwritten digit image. "
    "Answer with exactly one digit: "
    "0, 1, 2, 3, 4, 5, 6, 7, 8, or 9."
)

MODEL_NAME = "Moondream2-DoCCI-Instruct-MNIST"
COLLECTION_MODE = "automated_edge"


# ============================================================
# 2. OPTIONAL PACKAGES / HARDWARE METADATA
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

    print(
        "CodeCarbon unavailable. "
        "Energy values will be 0."
    )


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


try:
    import cpuinfo

    _CPU_INFO = cpuinfo.get_cpu_info()

    CPU_MODEL_NAME = _CPU_INFO.get(
        "brand_raw",
        "Unknown",
    )

    CPU_ARCH = _CPU_INFO.get(
        "arch",
        platform.machine(),
    )

except Exception:
    CPU_MODEL_NAME = (
        platform.processor()
        or "Unknown"
    )

    CPU_ARCH = platform.machine()


CPU_TDP_W = None

PYTHON_VERSION = sys.version.split()[0]
TORCH_VERSION = torch.__version__
TRANSFORMERS_VERSION = transformers.__version__

OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()

SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total
    / (1024 ** 3),
    2,
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


def get_os_full_name() -> str:

    system = platform.system()

    if system == "Windows":

        return (
            f"Windows "
            f"{platform.release()} "
            f"{platform.machine()}"
        )

    if system == "Linux":

        try:

            info = {}

            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    if "=" in line:

                        key, value = (
                            line
                            .strip()
                            .split("=", 1)
                        )

                        info[key] = (
                            value.strip('"')
                        )

            return (
                f"{info.get('PRETTY_NAME', 'Linux')} "
                f"{platform.machine()}"
            )

        except Exception:

            return (
                f"Linux "
                f"{platform.release()} "
                f"{platform.machine()}"
            )

    if system == "Darwin":

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


def make_stable_device_id() -> str:

    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


DEVICE_UUID = (
    make_stable_device_id()
)

DEVICE_SHORT = (
    DEVICE_UUID[:8]
)


def get_hostname() -> str:

    return socket.gethostname()


def get_cuda_driver_version():

    if not NVML_AVAILABLE:

        return None

    try:

        version = (
            pynvml
            .nvmlSystemGetDriverVersion()
        )

        if isinstance(
            version,
            bytes,
        ):
            return version.decode(
                "utf-8"
            )

        return version

    except Exception:

        return None


CUDA_DRIVER_VERSION = (
    get_cuda_driver_version()
)


def get_gpu_static() -> dict:

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

            props = (
                torch.cuda
                .get_device_properties(0)
            )

            result[
                "gpu_compute_capability"
            ] = (
                f"{props.major}."
                f"{props.minor}"
            )

        except Exception:

            pass


    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
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


def get_gpu_core_thread():

    if not torch.cuda.is_available():

        return None, None


    try:

        props = (
            torch.cuda
            .get_device_properties(0)
        )

        sm_count = (
            props.multi_processor_count
        )

        cores_per_sm = {
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(
            props.major,
            64,
        )

        gpu_core_count = (
            sm_count
            * cores_per_sm
        )

        gpu_thread_count = (
            sm_count
            * props
            .max_threads_per_multi_processor
        )

        return (
            gpu_core_count,
            gpu_thread_count,
        )

    except Exception:

        return None, None


(
    GPU_CORE_COUNT,
    GPU_THREAD_COUNT,
) = get_gpu_core_thread()


def get_gpu_name() -> str:

    try:

        if torch.cuda.is_available():

            return (
                torch.cuda
                .get_device_name(0)
            )

    except Exception:

        return "Unknown GPU"

    return "No GPU"


def get_cpu_usage():

    try:

        return psutil.cpu_percent(
            interval=None
        )

    except Exception:

        return None


def get_cpu_freq():

    try:

        freq = psutil.cpu_freq()

        return (
            round(
                freq.current,
                2,
            )
            if freq
            else None
        )

    except Exception:

        return None


def get_cpu_temp():

    try:

        temps = (
            psutil
            .sensors_temperatures()
        )

        if not temps:

            return None


        for key in (
            "coretemp",
            "k10temp",
            "cpu_thermal",
            "acpitz",
        ):

            if key in temps:

                values = [
                    item.current
                    for item
                    in temps[key]
                    if (
                        item.current
                        is not None
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

    # True CPU package power requires
    # platform-specific sensor access.
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


def get_memory_footprint_mb():

    try:

        return round(
            psutil
            .Process(
                os.getpid()
            )
            .memory_info()
            .rss
            / (1024 ** 2),
            4,
        )

    except Exception:

        return None


def get_gpu_metrics() -> dict:

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
                pynvml
                .NVML_TEMPERATURE_GPU,
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


        return {
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

    except Exception:

        return result


# ============================================================
# 3. TRANSFORMERS COMPATIBILITY PATCHES
# ============================================================

# This older Moondream instruct revision expects parts of the
# earlier Hugging Face cache API. These patches are harmless
# when the installed Transformers version already provides them.

if not hasattr(
    DynamicCache,
    "get_usable_length",
):

    def _get_usable_length(
        self,
        new_seq_length: int,
        layer_idx: int = 0,
    ) -> int:

        try:

            return int(
                self.get_seq_length(
                    layer_idx
                )
            )

        except TypeError:

            return int(
                self.get_seq_length()
            )


    DynamicCache.get_usable_length = (
        _get_usable_length
    )


if not hasattr(
    DynamicCache,
    "get_max_length",
):

    def _get_max_length(
        self
    ):

        return None


    DynamicCache.get_max_length = (
        _get_max_length
    )


# ============================================================
# 4. DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

    if (
        hasattr(
            torch.cuda,
            "is_bf16_supported",
        )
        and torch.cuda
        .is_bf16_supported()
    ):

        DTYPE = torch.bfloat16

    else:

        DTYPE = torch.float16

else:

    DEVICE = torch.device(
        "cpu"
    )

    DTYPE = torch.float32


print("=" * 72)
print("TEST CONFIGURATION")
print("=" * 72)

print(
    "Transformers:",
    TRANSFORMERS_VERSION,
)

print(
    "Model:",
    MODEL_ID,
)

print(
    "Revision:",
    MODEL_REVISION,
)

print(
    "Device:",
    DEVICE,
)

print(
    "Data type:",
    DTYPE,
)


if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda
        .get_device_name(0),
    )


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
)

print(
    "Tokenizer loaded."
)


# ============================================================
# 6. LOAD DoCCI INSTRUCT MODEL DIRECTLY
# ============================================================

print(
    "\nLoading "
    "fal/moondream2-docci-instruct "
    "directly..."
)

# ------------------------------------------------------------
# IMPORTANT FIX
# ------------------------------------------------------------
#
# The pinned DoCCI checkpoint contains its own matching Moondream
# implementation. Some config variants can redirect AutoModel to a
# newer vikhyatk/moondream2 implementation with a different vision
# width. That causes size mismatches such as:
#
#   checkpoint: [8192, 1152]
#   loaded model: [8192, 2304]
#
# Therefore, load the pinned DoCCI config and force the AutoModel
# class to come from moondream.py in THIS SAME pinned repository.
# ------------------------------------------------------------

config = AutoConfig.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
)

if getattr(config, "auto_map", None) is None:
    config.auto_map = {}

config.auto_map["AutoModelForCausalLM"] = (
    "moondream.Moondream"
)

print(
    "Forced AutoModel implementation:",
    config.auto_map["AutoModelForCausalLM"],
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        trust_remote_code=True,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
)

# ============================================================
# 7. GENERATION COMPATIBILITY PATCH
# ============================================================

def patch_generation_support(
    moondream_model,
) -> None:

    """
    Older Moondream repositories use a custom Phi model.
    Under newer Transformers versions, the text model may
    not expose .generate() through inheritance.

    If .generate() is missing, add GenerationMixin.
    """

    text_model = (
        moondream_model.text_model
    )


    if callable(
        getattr(
            text_model,
            "generate",
            None,
        )
    ):

        print(
            "Generation patch not needed."
        )

        return


    original_class = (
        text_model.__class__
    )


    patched_class = type(
        (
            f"{original_class.__name__}"
            f"WithGeneration"
        ),
        (
            GenerationMixin,
            original_class,
        ),
        {
            "__module__":
                original_class.__module__,
        },
    )


    text_model.__class__ = (
        patched_class
    )


    if not callable(
        getattr(
            text_model,
            "generate",
            None,
        )
    ):

        raise RuntimeError(
            "GenerationMixin patch "
            "did not add generate()."
        )


    if getattr(
        text_model,
        "generation_config",
        None,
    ) is None:

        text_model.generation_config = (
            GenerationConfig
            .from_model_config(
                text_model.config
            )
        )


    if not hasattr(
        text_model,
        "_supports_cache_class",
    ):

        text_model._supports_cache_class = (
            False
        )


    print(
        "Applied GenerationMixin "
        "compatibility patch."
    )


patch_generation_support(
    model
)


# ============================================================
# 7B. PATCH NEW TRANSFORMERS GENERATION KWARGS
# ============================================================

# Newer Transformers GenerationMixin passes cache_position into
# model.forward(). This historical PhiForCausalLM implementation
# predates that keyword and raises:
#
#   TypeError:
#   PhiForCausalLM.forward() got an unexpected keyword argument
#   'cache_position'
#
# Remove only unsupported generation-only kwargs before delegating
# to the original historical Phi forward method.

import inspect
import types


_original_phi_forward = model.text_model.forward

_phi_forward_parameters = set(
    inspect.signature(
        _original_phi_forward
    ).parameters.keys()
)


def _phi_forward_compat(
    self,
    *args,
    **kwargs,
):
    # Strip kwargs introduced by newer Transformers only when
    # the historical forward signature does not support them.
    for key in (
        "cache_position",
        "num_logits_to_keep",
    ):
        if (
            key in kwargs
            and key not in _phi_forward_parameters
        ):
            kwargs.pop(
                key,
                None,
            )

    return _original_phi_forward(
        *args,
        **kwargs,
    )


model.text_model.forward = types.MethodType(
    _phi_forward_compat,
    model.text_model,
)

print(
    "Applied PhiForCausalLM forward compatibility patch."
)


model.to(
    DEVICE
)

model.eval()


MODEL_PARAMETERS = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


# For this multimodal generative model, a single meaningful
# FLOPs value depends on image tokenization, prompt length,
# and generated token length. Leave as None rather than report
# a misleading number.
MODEL_FLOPS = None


print(
    "\nModel ready."
)

print(
    "Model parameters:",
    f"{MODEL_PARAMETERS:,}",
)


# ============================================================
# 8. LOAD MNIST TEST SET
# ============================================================

test_dataset = MNIST(
    root=MNIST_ROOT,
    train=False,
    download=True,

    # Keep transform=None so MNIST returns PIL images.
    transform=None,
)


if MAX_TEST_SAMPLES is None:

    number_of_samples = (
        len(test_dataset)
    )

else:

    if (
        not isinstance(
            MAX_TEST_SAMPLES,
            int,
        )
        or MAX_TEST_SAMPLES <= 0
    ):

        raise ValueError(
            "MAX_TEST_SAMPLES must "
            "be a positive integer "
            "or None."
        )


    number_of_samples = min(
        MAX_TEST_SAMPLES,
        len(test_dataset),
    )


print(
    "\nNumber of MNIST test images:",
    number_of_samples,
)


# ============================================================
# 9. NORMALIZE MODEL RESPONSE
# ============================================================

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def normalize_prediction(
    response: str,
) -> str:

    text = (
        str(response)
        .lower()
        .strip()
    )


    # Prefer an explicitly written digit.
    digit_matches = re.findall(
        r"(?<!\d)[0-9](?!\d)",
        text,
    )

    digit_matches = list(
        dict.fromkeys(
            digit_matches
        )
    )


    if len(digit_matches) == 1:

        return digit_matches[0]


    # Also accept number words.
    word_matches = []

    for word, digit in (
        NUMBER_WORDS.items()
    ):

        if re.search(
            rf"\b{word}\b",
            text,
        ):

            word_matches.append(
                digit
            )


    word_matches = list(
        dict.fromkeys(
            word_matches
        )
    )


    # If one unique number appears
    # across numeric and word forms,
    # accept it.
    combined = list(
        dict.fromkeys(
            digit_matches
            + word_matches
        )
    )


    if len(combined) == 1:

        return combined[0]


    return "invalid"


# ============================================================
# 10. TOKEN / CLASS-SCORING HELPERS
# ============================================================

def get_token_embedding_layer(
    moondream_model,
):

    text_model = (
        moondream_model.text_model
    )


    if hasattr(
        text_model,
        "get_input_embeddings",
    ):

        layer = (
            text_model
            .get_input_embeddings()
        )

        if layer is not None:

            return layer


    possible_paths = [
        lambda m:
            m.text_model
            .transformer
            .embd
            .wte,

        lambda m:
            m.get_input_embeddings(),
    ]


    for getter in possible_paths:

        try:

            layer = getter(
                moondream_model
            )

            if layer is not None:

                return layer

        except (
            AttributeError,
            TypeError,
        ):

            continue


    raise RuntimeError(
        "Could not locate the "
        "token embedding layer."
    )


TOKEN_EMBEDDING = (
    get_token_embedding_layer(
        model
    )
)


def get_start_token_id() -> int:

    if (
        tokenizer.bos_token_id
        is not None
    ):

        return int(
            tokenizer.bos_token_id
        )


    if (
        tokenizer.eos_token_id
        is not None
    ):

        return int(
            tokenizer.eos_token_id
        )


    raise RuntimeError(
        "Tokenizer has no "
        "BOS or EOS token."
    )


def tokenize_text(
    text: str,
) -> torch.Tensor:

    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(
        DEVICE
    )


def score_candidate_classes(
    image,
) -> dict:

    """
    Score all ten MNIST labels using the same
    Moondream text model.

    This is used only to obtain:
      confidence_score
      logit_margin
      entropy

    Final prediction still comes from the instruct model's
    generated answer.
    """

    image = image.convert(
        "RGB"
    )


    with torch.inference_mode():

        image_embeddings = (
            model.encode_image(
                image
            )
        )


    # Match the repository's answer_question prompt style.
    prompt_text = (
        f"\n\nQuestion: "
        f"{QUESTION}"
        f"\n\nAnswer:"
    )


    start_ids = torch.tensor(
        [
            [
                get_start_token_id()
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )


    prompt_ids = tokenize_text(
        prompt_text
    )


    start_embeddings = (
        TOKEN_EMBEDDING(
            start_ids
        )
    )


    prompt_embeddings = (
        TOKEN_EMBEDDING(
            prompt_ids
        )
    )


    image_embeddings = (
        image_embeddings.to(
            device=DEVICE,
            dtype=start_embeddings.dtype,
        )
    )


    prefix_embeddings = torch.cat(
        [
            start_embeddings,
            image_embeddings,
            prompt_embeddings,
        ],
        dim=1,
    )


    prefix_length = (
        prefix_embeddings
        .shape[1]
    )


    class_scores = []

    class_token_counts = []


    for class_name in CLASS_NAMES:

        class_ids = tokenize_text(
            class_name
        )


        class_embeddings = (
            TOKEN_EMBEDDING(
                class_ids
            )
            .to(
                dtype=
                    prefix_embeddings.dtype
            )
        )


        full_embeddings = torch.cat(
            [
                prefix_embeddings,
                class_embeddings,
            ],
            dim=1,
        )


        attention_mask = torch.ones(
            full_embeddings.shape[:2],
            dtype=torch.long,
            device=DEVICE,
        )


        with torch.inference_mode():

            outputs = (
                model.text_model(
                    inputs_embeds=
                        full_embeddings,

                    attention_mask=
                        attention_mask,

                    use_cache=False,

                    return_dict=True,
                )
            )


        logits = outputs.logits


        token_log_probs = []

        flat_class_ids = (
            class_ids[0]
        )


        for (
            token_offset,
            token_id,
        ) in enumerate(
            flat_class_ids
        ):

            prediction_position = (
                prefix_length
                - 1
                + token_offset
            )


            log_probs = (
                torch.log_softmax(
                    logits[
                        0,
                        prediction_position,
                        :
                    ],
                    dim=-1,
                )
            )


            token_log_probs.append(
                log_probs[
                    int(
                        token_id.item()
                    )
                ]
            )


        mean_log_prob = (
            torch.stack(
                token_log_probs
            )
            .mean()
        )


        class_scores.append(
            mean_log_prob
        )


        class_token_counts.append(
            int(
                class_ids.shape[1]
            )
        )


    scores = (
        torch.stack(
            class_scores
        )
        .float()
    )


    probabilities = (
        torch.softmax(
            scores,
            dim=0,
        )
    )


    top2_scores = (
        torch.topk(
            scores,
            k=2,
        )
        .values
    )


    entropy = float(
        -(
            probabilities
            * torch.log(
                probabilities
                + 1e-12
            )
        )
        .sum()
        .item()
    )


    best_index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    input_tokens = int(
        start_ids.shape[1]
        + image_embeddings.shape[1]
        + prompt_ids.shape[1]
    )


    return {
        "class_probabilities":
            probabilities
            .detach()
            .cpu(),

        "class_scores":
            scores
            .detach()
            .cpu(),

        "top_class_index":
            best_index,

        "top_class":
            CLASS_NAMES[
                best_index
            ],

        "confidence_score":
            float(
                probabilities[
                    best_index
                ].item()
            ),

        "logit_margin":
            float(
                (
                    top2_scores[0]
                    - top2_scores[1]
                ).item()
            ),

        "entropy":
            entropy,

        "input_tokens":
            input_tokens,

        "class_token_counts":
            class_token_counts,
    }


# ============================================================
# 11. INFERENCE
# ============================================================

def classify_image(
    image,
) -> str:

    image = image.convert(
        "RGB"
    )


    with torch.inference_mode():

        encoded_image = (
            model.encode_image(
                image
            )
        )


        answer = (
            model.answer_question(
                encoded_image,
                QUESTION,
                tokenizer,
            )
        )


    return str(
        answer
    ).strip()


# ============================================================
# 12. WARM-UP
# ============================================================

print(
    "\nRunning warm-up inference..."
)

warmup_image, _ = (
    test_dataset[0]
)


try:

    warmup_response = (
        classify_image(
            warmup_image
        )
    )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    print(
        "Warm-up response:",
        warmup_response,
    )


    print(
        "Warm-up completed successfully."
    )


except Exception:

    import traceback

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REAL WARM-UP ERROR"
    )

    print(
        "=" * 72
    )

    traceback.print_exc()

    raise RuntimeError(
        "Warm-up inference failed. "
        "The complete original "
        "error is printed above."
    )


# ============================================================
# 13. PER-SAMPLE INFERENCE + QUALITY
# ============================================================

def run_sample_inference(
    image,
) -> dict:

    raw_response = (
        classify_image(
            image
        )
    )


    prediction = (
        normalize_prediction(
            raw_response
        )
    )


    quality = (
        score_candidate_classes(
            image
        )
    )


    if prediction in CLASS_NAMES:

        prediction_index = (
            CLASS_NAMES.index(
                prediction
            )
        )


        confidence_score = float(
            quality[
                "class_probabilities"
            ][
                prediction_index
            ].item()
        )


        output_tokens = len(
            tokenizer(
                prediction,
                add_special_tokens=False,
            ).input_ids
        )

    else:

        confidence_score = 0.0


        output_tokens = len(
            tokenizer(
                str(
                    raw_response
                ),
                add_special_tokens=False,
            ).input_ids
        )


    return {
        "raw_response":
            raw_response,

        "prediction":
            prediction,

        "confidence_score":
            confidence_score,

        "logit_margin":
            quality[
                "logit_margin"
            ],

        "entropy":
            quality[
                "entropy"
            ],

        "input_tokens":
            quality[
                "input_tokens"
            ],

        "output_tokens":
            max(
                int(
                    output_tokens
                ),
                1,
            ),

        "quality_top_class":
            quality[
                "top_class"
            ],
    }


# ============================================================
# 14. ENERGY TRACKING
# ============================================================

def run_with_energy_tracking(
    inference_fn,
    image,
):

    energy_dir = (
        OUTPUT_ROOT / "codecarbon"
    )

    energy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    if CODECARBON_AVAILABLE:

        tracker = (
            EmissionsTracker(
                project_name=(
                    "moondream_docci_"
                    "mnist_test"
                ),

                output_dir=str(
                    energy_dir
                ),

                output_file=(
                    "codecarbon_"
                    "moondream_docci_"
                    "mnist.csv"
                ),

                log_level="error",

                save_to_file=True,
            )
        )


        tracker.start()


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start_time = (
            time.perf_counter()
        )


        result = (
            inference_fn(
                image
            )
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        exec_time = (
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
            or 0
        )


        gpu_energy = float(
            getattr(
                final_data,
                "gpu_energy",
                0,
            )
            or 0
        )


        ram_energy = float(
            getattr(
                final_data,
                "ram_energy",
                0,
            )
            or 0
        )


        total_energy = float(
            getattr(
                final_data,
                "energy_consumed",
                0,
            )
            or 0
        )


        carbon_intensity = None


        if (
            emissions_value
            is not None
            and total_energy > 0
        ):

            carbon_intensity = (
                float(
                    emissions_value
                )
                / total_energy
            )


    else:

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start_time = (
            time.perf_counter()
        )


        result = (
            inference_fn(
                image
            )
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        exec_time = (
            time.perf_counter()
            - start_time
        )


        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None


    gpu_metrics = (
        get_gpu_metrics()
    )


    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        gpu_metrics,
    )


# ============================================================
# 15. BUILD TELEMETRY ROW
# ============================================================

def build_row(
    sample_index: int,
    true_label: str,
    prediction: str,
    correct: bool,
    result: dict,
    exec_time: float,
    cpu_energy: float,
    gpu_energy: float,
    ram_energy: float,
    total_energy: float,
    emissions_value,
    carbon_intensity,
    gpu_metrics: dict,
    error_message: str,
) -> dict:

    input_tokens = int(
        result.get(
            "input_tokens",
            0,
        )
        or 0
    )


    output_tokens = int(
        result.get(
            "output_tokens",
            0,
        )
        or 0
    )


    total_tokens = (
        input_tokens
        + output_tokens
    )


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
            / total_tokens
        )


        joules_per_token = (
            total_energy
            * 3_600_000
            / total_tokens
        )


        if exec_time > 0:

            watts_estimated = (
                total_energy
                * 3_600_000
                / exec_time
            )


        gpu_energy_pct = (
            gpu_energy
            / total_energy
            * 100
        )


        cpu_energy_pct = (
            cpu_energy
            / total_energy
            * 100
        )


    return {
        # --- Identity ---
        "timestamp":                    time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id":             DEVICE_UUID,
        "device_short_id":              DEVICE_SHORT,
        "pc_name":                      get_hostname(),
        "collection_mode":              "automated_edge",

        # --- Sample ---
        "sample_index":                 sample_index,
        "true_label":                   true_label,
        "prediction":                   prediction,
        "correct":                      correct,

        # --- Model identity ---
        "model_type":                   MODEL_NAME,
        "parameters":                   MODEL_PARAMETERS,
        "model_flops":                  MODEL_FLOPS,

        # --- Prediction quality ---
        "confidence_score":             (
            round(float(result["confidence_score"]), 6)
            if result.get("confidence_score") is not None
            else None
        ),
        "logit_margin":                 (
            round(float(result["logit_margin"]), 6)
            if result.get("logit_margin") is not None
            else None
        ),
        "entropy":                      (
            round(float(result["entropy"]), 6)
            if result.get("entropy") is not None
            else None
        ),

        # --- Timing ---
        "execution_time_sec":           round(exec_time, 10),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh":               cpu_energy,
        "gpu_energy_kwh":               gpu_energy,
        "ram_energy_kwh":               ram_energy,
        "total_energy_kwh":             total_energy,
        "total_emissions_kg":           emissions_value,
        "carbon_intensity_kgco2_kwh":   carbon_intensity,
        "codecarbon_version":           CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens":                 input_tokens,
        "output_tokens":                output_tokens,
        "total_tokens":                 total_tokens,
        "tokens_per_second":            (
            round(total_tokens / exec_time, 4)
            if exec_time > 0
            else None
        ),
        "joules_per_token":             round(joules_per_token, 8),
        "energy_per_token_kwh":         round(energy_per_token_kwh, 12),
        "watts_estimated":              round(watts_estimated, 4),
        "gpu_energy_pct_of_total":      round(gpu_energy_pct, 2),
        "cpu_energy_pct_of_total":      round(cpu_energy_pct, 2),

        # --- CPU hardware ---
        "cpu_model":                    CPU_MODEL_NAME,
        "cpu_architecture":             CPU_ARCH,
        "cpu_core_count":               CPU_CORE_COUNT,
        "cpu_thread_count":             CPU_THREAD_COUNT,
        "cpu_core":                     CPU_CORE_COUNT,
        "cpu_thread":                   CPU_THREAD_COUNT,
        "cpu_tdp_w":                    CPU_TDP_W,
        "cpu_usage_pct":                get_cpu_usage(),
        "cpu_clock_mhz":                get_cpu_freq(),
        "cpu_temp_c":                   get_cpu_temp(),
        "cpu_power_draw_w":             get_cpu_power_draw_w(),
        "cpu_cores_used":               get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model":                    get_gpu_name(),
        "gpu_core":                     GPU_CORE_COUNT,
        "gpu_thread":                   GPU_THREAD_COUNT,
        "gpu_driver_version":           GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability":       GPU_STATIC["gpu_compute_capability"],
        "gpu_power_limit_w":            GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb":          GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w":             gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct":          gpu_metrics.get("gpu_utilization_pct"),
        "gpu_temp_c":                   gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb":           gpu_metrics.get("gpu_memory_used_mb"),
        "gpu_sm_clock_mhz":             gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz":         gpu_metrics.get("gpu_memory_clock_mhz"),
        "cuda_driver_version":          CUDA_DRIVER_VERSION,
        "cuda_available":               torch.cuda.is_available(),
        "device_type":                  str(DEVICE),

        # --- RAM / memory ---
        "ram_usage_pct":                psutil.virtual_memory().percent,
        "memory_footprint_mb":          get_memory_footprint_mb(),
        "system_ram_total_gb":          SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name":                      OS_NAME,
        "os_version":                   OS_VERSION,
        "os_architecture":              OS_ARCHITECTURE,
        "os_full_name":                 OS_FULL_NAME,
        "python_version":               PYTHON_VERSION,
        "torch_version":                TORCH_VERSION,

        # --- Final model metrics (backfilled after run) ---
        "model_accuracy":               None,
        "model_precision_weighted":     None,
        "model_recall_weighted":        None,
        "model_f1_weighted":            None,
    }


# ============================================================
# 16. TEST LOOP
# ============================================================

results = []

true_labels = []

predicted_labels = []

latencies = []

inference_error_count = 0


print(
    "\n"
    + "=" * 72
)

print(
    "TESTING MOONDREAM2 DoCCI INSTRUCT ON MNIST"
)

print(
    "=" * 72
)


for test_index in tqdm(
    range(
        number_of_samples
    ),
    desc="Testing",
):

    image, true_label_id = (
        test_dataset[
            test_index
        ]
    )


    true_label_id = int(
        true_label_id
    )


    true_label = (
        CLASS_NAMES[
            true_label_id
        ]
    )


    result = {
        "raw_response":
            "",

        "prediction":
            "invalid",

        "confidence_score":
            None,

        "logit_margin":
            None,

        "entropy":
            None,

        "input_tokens":
            0,

        "output_tokens":
            0,

        "quality_top_class":
            "",
    }


    error_message = ""


    try:

        (
            result,
            exec_time,
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            emissions_value,
            carbon_intensity,
            gpu_metrics,
        ) = run_with_energy_tracking(
            run_sample_inference,
            image,
        )


    except Exception as error:

        error_message = repr(
            error
        )

        inference_error_count += 1


        exec_time = 0.0
        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None

        gpu_metrics = (
            get_gpu_metrics()
        )


    prediction = (
        result.get(
            "prediction",
            "invalid",
        )
    )


    correct = (
        prediction
        == true_label
    )


    true_labels.append(
        true_label
    )


    predicted_labels.append(
        prediction
    )


    latencies.append(
        float(
            exec_time
        )
    )


    row = build_row(
        sample_index=int(
            test_index
        ),

        true_label=
            true_label,

        prediction=
            prediction,

        correct=
            correct,

        result=
            result,

        exec_time=
            exec_time,

        cpu_energy=
            cpu_energy,

        gpu_energy=
            gpu_energy,

        ram_energy=
            ram_energy,

        total_energy=
            total_energy,

        emissions_value=
            emissions_value,

        carbon_intensity=
            carbon_intensity,

        gpu_metrics=
            gpu_metrics,

        error_message=
            error_message,
    )


    results.append(
        row
    )


    print(
        f"\nImage "
        f"{test_index + 1}/"
        f"{number_of_samples}"

        f"\nTrue label:  "
        f"{true_label}"

        f"\nPrediction:  "
        f"{prediction}"

        f"\nRaw answer:  "
        f"{result.get('raw_response', '')}"

        f"\nConfidence:  "
        f"{result.get('confidence_score')}"

        f"\nCorrect:     "
        f"{correct}"

        f"\nLatency:     "
        f"{exec_time:.4f} seconds"
    )


    if error_message:

        print(
            "Error:       ",
            error_message,
        )


# ============================================================
# 17. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)


precision_weighted = (
    precision_score(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        average=
            "weighted",

        zero_division=
            0,
    )
)


recall_weighted = (
    recall_score(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        average=
            "weighted",

        zero_division=
            0,
    )
)


f1_weighted = (
    f1_score(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        average=
            "weighted",

        zero_division=
            0,
    )
)


macro_f1 = (
    f1_score(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        average=
            "macro",

        zero_division=
            0,
    )
)


invalid_count = sum(
    prediction == "invalid"
    for prediction
    in predicted_labels
)


invalid_rate = (
    invalid_count
    / number_of_samples
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


report_dictionary = (
    classification_report(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        digits=
            4,

        zero_division=
            0,

        output_dict=
            True,
    )
)


matrix_labels = (
    CLASS_NAMES
    + ["invalid"]
)


matrix = (
    confusion_matrix(
        true_labels,
        predicted_labels,

        labels=
            matrix_labels,
    )
)


# ============================================================
# 18. BACKFILL FINAL METRICS
# ============================================================

for row in results:

    row[
        "model_accuracy"
    ] = float(
        accuracy
    )


    row[
        "model_precision_weighted"
    ] = float(
        precision_weighted
    )


    row[
        "model_recall_weighted"
    ] = float(
        recall_weighted
    )


    row[
        "model_f1_weighted"
    ] = float(
        f1_weighted
    )


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 72
)


print(
    f"Number of images:      "
    f"{number_of_samples}"
)

print(
    f"Accuracy:              "
    f"{accuracy:.4f}"
)

print(
    f"Weighted precision:    "
    f"{precision_weighted:.4f}"
)

print(
    f"Weighted recall:       "
    f"{recall_weighted:.4f}"
)

print(
    f"Weighted F1:           "
    f"{f1_weighted:.4f}"
)

print(
    f"Macro F1:              "
    f"{macro_f1:.4f}"
)

print(
    f"Invalid predictions:   "
    f"{invalid_count}"
)

print(
    f"Invalid rate:          "
    f"{invalid_rate:.4f}"
)

print(
    f"Inference errors:      "
    f"{inference_error_count}"
)

print(
    f"Average latency:       "
    f"{average_latency:.4f} seconds"
)

print(
    f"Median latency:        "
    f"{median_latency:.4f} seconds"
)

print(
    f"Minimum latency:       "
    f"{minimum_latency:.4f} seconds"
)

print(
    f"Maximum latency:       "
    f"{maximum_latency:.4f} seconds"
)


print(
    "\nClassification report:\n"
)


print(
    classification_report(
        true_labels,
        predicted_labels,

        labels=
            CLASS_NAMES,

        digits=
            4,

        zero_division=
            0,
    )
)


print(
    "\nConfusion-matrix "
    "label order:"
)

print(
    matrix_labels
)


print(
    "\nConfusion matrix:"
)

print(
    matrix
)


# ============================================================
# 20. SAVE PREDICTIONS / TELEMETRY
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


pd.DataFrame(
    results
).to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# 21. SAVE METRICS JSON
# ============================================================

metrics = {
    "transformers_version":
        TRANSFORMERS_VERSION,

    "model_id":
        MODEL_ID,

    "model_revision":
        MODEL_REVISION,

    "model_type":
        MODEL_NAME,

    "parameters":
        int(
            MODEL_PARAMETERS
        ),

    "model_flops":
        MODEL_FLOPS,

    "dataset":
        "MNIST",

    "number_of_samples":
        int(
            number_of_samples
        ),

    "device":
        str(
            DEVICE
        ),

    "dtype":
        str(
            DTYPE
        ),

    "accuracy":
        float(
            accuracy
        ),

    "precision_weighted":
        float(
            precision_weighted
        ),

    "recall_weighted":
        float(
            recall_weighted
        ),

    "f1_weighted":
        float(
            f1_weighted
        ),

    "macro_f1":
        float(
            macro_f1
        ),

    "invalid_count":
        int(
            invalid_count
        ),

    "invalid_rate":
        float(
            invalid_rate
        ),

    "inference_error_count":
        int(
            inference_error_count
        ),

    "average_latency_seconds":
        average_latency,

    "median_latency_seconds":
        median_latency,

    "minimum_latency_seconds":
        minimum_latency,

    "maximum_latency_seconds":
        maximum_latency,

    "class_names":
        CLASS_NAMES,

    "confusion_matrix_labels":
        matrix_labels,

    "confusion_matrix":
        matrix.tolist(),

    "classification_report":
        report_dictionary,
}





print(
    "\nSaved predictions / telemetry:"
)

print(
    OUTPUT_CSV.resolve()
)