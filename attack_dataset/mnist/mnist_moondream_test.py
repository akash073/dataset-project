from __future__ import annotations

import csv
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
from PIL import Image
from safetensors.torch import load_file
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
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    DynamicCache,
)
from transformers.generation.utils import GenerationMixin


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"

# Folder produced by your training code.
SAVED_MODEL_DIR = Path(
    "./moondream_mnist_finetuned/final_model"
)

MNIST_ROOT = "./data"

OUTPUT_CSV = Path(
    "./moondream_mnist_test_predictions.csv"
)

METRICS_JSON = Path(
    "./moondream_mnist_test_metrics.json"
)

# Set to None to test all 10,000 MNIST test images.
MAX_TEST_SAMPLES = 10

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

# Keep the question identical to training.
QUESTION = (
    "Classify this MNIST handwritten digit image. "
    "Answer with exactly one digit: "
    "0, 1, 2, 3, 4, 5, 6, 7, 8, or 9."
)



# ============================================================
# 2. HARDWARE / ENVIRONMENT / ENERGY METADATA
# ============================================================

MODEL_NAME = "Moondream2-MNIST-Finetuned"
COLLECTION_MODE = "automated_edge"

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = codecarbon.__version__
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"
    print("CodeCarbon unavailable. Energy values will be 0.")

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
    CPU_MODEL_NAME = _CPU_INFO.get("brand_raw", "Unknown")
    CPU_ARCH = _CPU_INFO.get("arch", platform.machine())
except Exception:
    CPU_MODEL_NAME = platform.processor() or "Unknown"
    CPU_ARCH = platform.machine()

CPU_TDP_W = None

PYTHON_VERSION = sys.version.split()[0]
TORCH_VERSION = torch.__version__
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total / (1024 ** 3),
    2,
)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)


def get_os_full_name() -> str:
    system = platform.system()

    if system == "Windows":
        return f"Windows {platform.release()} {platform.machine()}"

    if system == "Linux":
        try:
            info = {}
            with open("/etc/os-release", "r", encoding="utf-8") as file:
                for line in file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        info[key] = value.strip('"')
            return (
                f"{info.get('PRETTY_NAME', 'Linux')} "
                f"{platform.machine()}"
            )
        except Exception:
            return f"Linux {platform.release()} {platform.machine()}"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {platform.machine()}"

    return f"{system} {platform.release()} {platform.machine()}"


OS_FULL_NAME = get_os_full_name()


def make_stable_device_id() -> str:
    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]


def get_hostname() -> str:
    return socket.gethostname()


def get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode("utf-8") if isinstance(version, bytes) else version
    except Exception:
        return None


CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static() -> dict:
    result = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
    }

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            result["gpu_compute_capability"] = (
                f"{props.major}.{props.minor}"
            )
        except Exception:
            pass

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(
            NVML_HANDLE
        )
        memory = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)

        result["gpu_power_limit_w"] = round(
            power_limit_mw / 1000.0,
            2,
        )
        result["gpu_memory_total_mb"] = round(
            memory.total / (1024 ** 2),
            2,
        )
    except Exception:
        pass

    return result


GPU_STATIC = get_gpu_static()


def get_gpu_core_thread():
    if not torch.cuda.is_available():
        return None, None

    try:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
        cores_per_sm = {
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(props.major, 64)

        gpu_core_count = sm_count * cores_per_sm
        gpu_thread_count = (
            sm_count * props.max_threads_per_multi_processor
        )
        return gpu_core_count, gpu_thread_count
    except Exception:
        return None, None


GPU_CORE_COUNT, GPU_THREAD_COUNT = get_gpu_core_thread()


def get_gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"
    return "No GPU"


def get_cpu_usage():
    return psutil.cpu_percent(interval=None)


def get_cpu_freq():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                values = [
                    item.current
                    for item in temps[key]
                    if item.current is not None
                ]
                if values:
                    return round(sum(values) / len(values), 2)
    except Exception:
        pass

    return None


def get_cpu_power_draw_w():
    # Platform-specific sensor support is required for a true value.
    return None


def get_cpu_cores_used():
    try:
        return sum(
            1
            for value in psutil.cpu_percent(percpu=True)
            if value > 1.0
        )
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        return round(
            psutil.Process(os.getpid()).memory_info().rss
            / (1024 ** 2),
            4,
        )
    except Exception:
        return None


def get_gpu_metrics() -> dict:
    result = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
    }

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(
            NVML_HANDLE
        )
        temperature = pynvml.nvmlDeviceGetTemperature(
            NVML_HANDLE,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        memory = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        sm_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_SM,
        )
        memory_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_MEM,
        )

        return {
            "gpu_power_draw_w": round(power_mw / 1000.0, 2),
            "gpu_utilization_pct": utilization.gpu,
            "gpu_temp_c": temperature,
            "gpu_memory_used_mb": round(
                memory.used / (1024 ** 2),
                2,
            ),
            "gpu_sm_clock_mhz": sm_clock,
            "gpu_memory_clock_mhz": memory_clock,
        }
    except Exception:
        return result


# ============================================================
# 2. TRANSFORMERS VERSION CHECK
# ============================================================

REQUIRED_TRANSFORMERS_VERSION = "4.56.1"

if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
    raise RuntimeError(
        "\nIncorrect Transformers version.\n"
        f"Installed: {transformers.__version__}\n"
        f"Required:  {REQUIRED_TRANSFORMERS_VERSION}\n\n"
        "Install the required version with:\n"
        "pip uninstall transformers tokenizers -y\n"
        "pip install transformers==4.56.1\n"
    )



# ============================================================
# 3. PATCH TRANSFORMERS 4.56.1 CACHE API
# ============================================================

def patch_dynamic_cache_api() -> None:
    """
    Moondream revision 2024-08-26 expects the older Hugging Face
    Cache API, including get_usable_length() and get_max_length().

    Transformers 4.56.1 uses get_seq_length() and newer cache
    abstractions. These compatibility methods let the historical
    Moondream modeling_phi.py work without modifying the cached
    remote-code file.
    """

    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(
            self,
            new_seq_length: int,
            layer_idx: int = 0,
        ) -> int:
            try:
                return int(self.get_seq_length(layer_idx))
            except TypeError:
                return int(self.get_seq_length())

        DynamicCache.get_usable_length = get_usable_length

        print(
            "Patched DynamicCache.get_usable_length() "
            "for Transformers 4.56.1."
        )

    if not hasattr(DynamicCache, "get_max_length"):

        def get_max_length(self):
            # DynamicCache has no fixed maximum length.
            return None

        DynamicCache.get_max_length = get_max_length

        print(
            "Patched DynamicCache.get_max_length() "
            "for Transformers 4.56.1."
        )


patch_dynamic_cache_api()


# ============================================================
# 4. CHECK SAVED MODEL
# ============================================================

if not SAVED_MODEL_DIR.exists():
    raise FileNotFoundError(
        "Saved model directory was not found:\n"
        f"{SAVED_MODEL_DIR.resolve()}"
    )

weight_candidates = [
    SAVED_MODEL_DIR / "model.safetensors",
    SAVED_MODEL_DIR / "pytorch_model.bin",
    SAVED_MODEL_DIR / "model.safetensors.index.json",
    SAVED_MODEL_DIR / "pytorch_model.bin.index.json",
]

if not any(path.exists() for path in weight_candidates):
    raise FileNotFoundError(
        "No saved model weights were found in:\n"
        f"{SAVED_MODEL_DIR.resolve()}"
    )


# ============================================================
# 4. DEVICE CONFIGURATION
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

print("=" * 72)
print("TEST CONFIGURATION")
print("=" * 72)
print("Transformers:", transformers.__version__)
print("Base model:", MODEL_ID)
print("Base revision:", MODEL_REVISION)
print("Saved model:", SAVED_MODEL_DIR.resolve())
print("Device:", DEVICE)
print("Data type:", DTYPE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        SAVED_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("Tokenizer loaded from saved model directory.")

except Exception as error:
    print("Local tokenizer load failed:", repr(error))
    print("Loading tokenizer from the pinned base revision...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )


# ============================================================
# 6. LOAD BASE ARCHITECTURE
# ============================================================

print("\nLoading Moondream base architecture...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

print("Base architecture loaded.")


# ============================================================
# 7. LOAD SAVED FINE-TUNED WEIGHTS
# ============================================================

def load_checkpoint_state_dict(
    model_directory: Path,
) -> dict[str, torch.Tensor]:

    safe_file = model_directory / "model.safetensors"
    torch_file = model_directory / "pytorch_model.bin"

    safe_index = (
        model_directory / "model.safetensors.index.json"
    )
    torch_index = (
        model_directory / "pytorch_model.bin.index.json"
    )

    if safe_file.exists():
        print("Loading:", safe_file.resolve())
        return load_file(
            str(safe_file),
            device="cpu",
        )

    if torch_file.exists():
        print("Loading:", torch_file.resolve())
        return torch.load(
            torch_file,
            map_location="cpu",
            weights_only=True,
        )

    if safe_index.exists():
        print("Loading sharded SafeTensors checkpoint...")

        with safe_index.open(
            "r",
            encoding="utf-8",
        ) as file:
            index_data = json.load(file)

        state_dict = {}

        shard_names = sorted(
            set(index_data["weight_map"].values())
        )

        for shard_name in shard_names:
            shard_path = model_directory / shard_name
            print("Loading shard:", shard_path.name)

            state_dict.update(
                load_file(
                    str(shard_path),
                    device="cpu",
                )
            )

        return state_dict

    if torch_index.exists():
        print("Loading sharded PyTorch checkpoint...")

        with torch_index.open(
            "r",
            encoding="utf-8",
        ) as file:
            index_data = json.load(file)

        state_dict = {}

        shard_names = sorted(
            set(index_data["weight_map"].values())
        )

        for shard_name in shard_names:
            shard_path = model_directory / shard_name
            print("Loading shard:", shard_path.name)

            state_dict.update(
                torch.load(
                    shard_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        return state_dict

    raise FileNotFoundError(
        "No supported checkpoint file was found."
    )


checkpoint = load_checkpoint_state_dict(
    SAVED_MODEL_DIR
)

load_result = model.load_state_dict(
    checkpoint,
    strict=False,
)

missing_keys = list(load_result.missing_keys)
unexpected_keys = list(load_result.unexpected_keys)

del checkpoint

print("\nFine-tuned weights loaded.")
print("Missing keys:", len(missing_keys))
print("Unexpected keys:", len(unexpected_keys))

if missing_keys:
    print("\nMissing keys:")
    for key in missing_keys[:20]:
        print(" ", key)

if unexpected_keys:
    print("\nUnexpected keys:")
    for key in unexpected_keys[:20]:
        print(" ", key)

if len(missing_keys) > 20 or len(unexpected_keys) > 20:
    raise RuntimeError(
        "\nThe checkpoint has too many mismatched keys.\n"
        "Confirm that training used:\n"
        f"MODEL_ID = {MODEL_ID!r}\n"
        f"MODEL_REVISION = {MODEL_REVISION!r}"
    )


# ============================================================
# 8. PATCH GENERATION FOR TRANSFORMERS 4.56.1
# ============================================================

def patch_generation_support(moondream_model) -> None:
    """
    Moondream revision 2024-08-26 uses an older PhiForCausalLM
    implementation. Under Transformers 4.56.1, that class may no
    longer inherit GenerationMixin automatically.

    This patch adds GenerationMixin to the loaded text-model class.
    """

    text_model = moondream_model.text_model

    if callable(getattr(text_model, "generate", None)):
        print(
            "Generation patch not needed: "
            "text_model.generate() already exists."
        )
        return

    original_class = text_model.__class__

    # Put GenerationMixin first so its generate() implementation
    # takes precedence in the method-resolution order.
    patched_class = type(
        f"{original_class.__name__}WithGeneration",
        (GenerationMixin, original_class),
        {
            "__module__": original_class.__module__,
        },
    )

    text_model.__class__ = patched_class

    if not callable(getattr(text_model, "generate", None)):
        raise RuntimeError(
            "GenerationMixin patch did not add generate()."
        )

    if getattr(text_model, "generation_config", None) is None:
        text_model.generation_config = (
            GenerationConfig.from_model_config(
                text_model.config
            )
        )

    # GenerationMixin uses this property in newer releases.
    if not hasattr(text_model, "_supports_cache_class"):
        text_model._supports_cache_class = False

    print(
        "Applied GenerationMixin compatibility patch for "
        f"Transformers {transformers.__version__}."
    )


patch_generation_support(model)

model.to(DEVICE)
model.eval()

MODEL_PARAMETERS = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# A defensible FLOPs figure for this multimodal generation path requires
# fixing visual-token count, prompt length, and generated sequence length.
MODEL_FLOPS = None

print("\nFine-tuned model is ready.")
print("Model parameters:", f"{MODEL_PARAMETERS:,}")


# ============================================================
# 9. LOAD MNIST TEST SET
# ============================================================

test_dataset = MNIST(
    root=MNIST_ROOT,
    train=False,
    download=False,
)

if MAX_TEST_SAMPLES is None:
    number_of_samples = len(test_dataset)
else:
    if (
        not isinstance(MAX_TEST_SAMPLES, int)
        or MAX_TEST_SAMPLES <= 0
    ):
        raise ValueError(
            "MAX_TEST_SAMPLES must be a positive integer "
            "or None."
        )

    number_of_samples = min(
        MAX_TEST_SAMPLES,
        len(test_dataset),
    )

print("\nNumber of MNIST test images:", number_of_samples)


# ============================================================
# 10. NORMALIZE PREDICTION
# ============================================================

def normalize_prediction(response: str) -> str:
    """
    Normalize Moondream's generated answer to one MNIST class: "0"..."9".

    Examples:
        "7" -> "7"
        "The digit is 7." -> "7"
        "digit: 3" -> "3"

    Returns "invalid" if the response does not contain exactly one
    unambiguous MNIST digit class.
    """
    text = str(response).strip()

    matches = re.findall(r"(?<!\\d)([0-9])(?!\\d)", text)
    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) == 1:
        return unique_matches[0]

    return "invalid"


# ============================================================
# 11. PREDICTION-QUALITY / TOKEN HELPERS
# ============================================================

def find_token_embedding_layer(moondream_model):
    possible_paths = [
        lambda m: m.text_model.transformer.embd.wte,
        lambda m: m.text_model.get_input_embeddings(),
        lambda m: m.get_input_embeddings(),
    ]

    for getter in possible_paths:
        try:
            layer = getter(moondream_model)
            if layer is not None:
                return layer
        except (AttributeError, TypeError):
            continue

    raise RuntimeError(
        "Could not locate Moondream's token embedding layer."
    )


TOKEN_EMBEDDING = find_token_embedding_layer(model)


def get_start_token_id() -> int:
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id

    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id

    raise RuntimeError("Tokenizer has no BOS or EOS token.")


def extract_tensor_from_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in (
            "image_embeds",
            "image_embeddings",
            "embeddings",
            "last_hidden_state",
        ):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(
        "Unsupported vision-encoder output type: "
        f"{type(output)}"
    )


def encode_image_for_scoring(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")

    processed = model.vision_encoder.preprocess(image)

    if isinstance(processed, list):
        processed = torch.stack(processed)

    if processed.ndim == 3:
        processed = processed.unsqueeze(0)

    processed = processed.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    with torch.inference_mode():
        try:
            output = model.vision_encoder(processed)
        except TypeError:
            output = model.vision_encoder(
                pixel_values=processed
            )

    embeddings = extract_tensor_from_output(output)

    if embeddings.ndim == 2:
        embeddings = embeddings.unsqueeze(0)

    return embeddings


def tokenize_text(text: str) -> torch.Tensor:
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(DEVICE)


def score_candidate_classes(image: Image.Image) -> dict:
    """
    Compute a normalized probability distribution over the 10 MNIST
    digit classes (0-9).

    Each class can contain more than one tokenizer token. We therefore
    score the complete class string and use mean token log-probability
    so longer class names are not unfairly penalized.
    """

    image_embeddings = encode_image_for_scoring(image)

    prompt_text = (
        f"\\n\\nQuestion: {QUESTION}"
        "\\n\\nAnswer:"
    )

    start_ids = torch.tensor(
        [[get_start_token_id()]],
        dtype=torch.long,
        device=DEVICE,
    )
    prompt_ids = tokenize_text(prompt_text)

    start_embeddings = TOKEN_EMBEDDING(start_ids)
    prompt_embeddings = TOKEN_EMBEDDING(prompt_ids)

    image_embeddings = image_embeddings.to(
        device=DEVICE,
        dtype=start_embeddings.dtype,
    )

    prefix_embeddings = torch.cat(
        [
            start_embeddings,
            image_embeddings,
            prompt_embeddings,
        ],
        dim=1,
    )

    prefix_length = prefix_embeddings.shape[1]

    class_scores = []
    class_token_counts = []

    for class_name in CLASS_NAMES:
        class_ids = tokenize_text(class_name)
        class_embeddings = TOKEN_EMBEDDING(class_ids).to(
            dtype=prefix_embeddings.dtype
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
            outputs = model.text_model(
                inputs_embeds=full_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        logits = outputs.logits

        token_log_probs = []
        flat_class_ids = class_ids[0]

        for token_offset, token_id in enumerate(flat_class_ids):
            prediction_position = prefix_length - 1 + token_offset
            log_probs = torch.log_softmax(
                logits[0, prediction_position, :],
                dim=-1,
            )
            token_log_probs.append(
                log_probs[int(token_id.item())]
            )

        mean_log_prob = torch.stack(
            token_log_probs
        ).mean()

        class_scores.append(mean_log_prob)
        class_token_counts.append(
            int(class_ids.shape[1])
        )

    scores = torch.stack(class_scores).float()
    probabilities = torch.softmax(scores, dim=0)

    top2_scores = torch.topk(scores, k=2).values

    entropy = float(
        -(
            probabilities
            * torch.log(probabilities + 1e-12)
        ).sum().item()
    )

    best_index = int(
        torch.argmax(probabilities).item()
    )

    input_tokens = int(
        start_ids.shape[1]
        + image_embeddings.shape[1]
        + prompt_ids.shape[1]
    )

    return {
        "class_probabilities": probabilities.detach().cpu(),
        "class_scores": scores.detach().cpu(),
        "top_class_index": best_index,
        "top_class": CLASS_NAMES[best_index],
        "confidence_score": float(
            probabilities[best_index].item()
        ),
        "logit_margin": float(
            (top2_scores[0] - top2_scores[1]).item()
        ),
        "entropy": entropy,
        "input_tokens": input_tokens,
        "class_token_counts": class_token_counts,
    }


# ============================================================
# 11. INFERENCE
# ============================================================

def classify_image(image: Image.Image) -> str:
    image = image.convert("RGB")

    with torch.inference_mode():
        if (
            hasattr(model, "encode_image")
            and hasattr(model, "answer_question")
        ):
            encoded_image = model.encode_image(image)

            answer = model.answer_question(
                encoded_image,
                QUESTION,
                tokenizer,
            )

            return str(answer).strip()

        if hasattr(model, "query"):
            result = model.query(
                image=image,
                question=QUESTION,
            )

            if isinstance(result, dict):
                return str(
                    result.get("answer", "")
                ).strip()

            return str(result).strip()

    raise AttributeError(
        "The loaded model has no compatible inference API."
    )


# ============================================================
# 12. WARM-UP
# ============================================================

print("\nRunning warm-up inference...")

warmup_image, _ = test_dataset[0]

try:
    warmup_response = classify_image(warmup_image)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print("Warm-up response:", warmup_response)
    print("Warm-up completed successfully.")

except Exception:
    import traceback

    print("\n" + "=" * 72)
    print("REAL WARM-UP ERROR")
    print("=" * 72)
    traceback.print_exc()

    raise RuntimeError(
        "\nWarm-up inference failed. The complete original error "
        "is printed directly above this message."
    )


# ============================================================
# 13. ENERGY-TRACKED TEST ONE IMAGE AT A TIME
# ============================================================

def run_sample_inference(image: Image.Image) -> dict:
    """
    Run the original Moondream answer-generation path and separately
    score the ten valid MNIST labels for confidence statistics.
    """

    raw_response = classify_image(image)
    prediction = normalize_prediction(raw_response)
    quality = score_candidate_classes(image)

    if prediction in CLASS_NAMES:
        prediction_index = CLASS_NAMES.index(prediction)
        confidence_score = float(
            quality["class_probabilities"][
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
                str(raw_response),
                add_special_tokens=False,
            ).input_ids
        )

    return {
        "raw_response": raw_response,
        "prediction": prediction,
        "confidence_score": confidence_score,
        "logit_margin": quality["logit_margin"],
        "entropy": quality["entropy"],
        "input_tokens": quality["input_tokens"],
        "output_tokens": max(int(output_tokens), 1),
        "quality_top_class": quality["top_class"],
    }


def run_with_energy_tracking(
    inference_fn,
    image: Image.Image,
):
    energy_dir = OUTPUT_CSV.parent / "codecarbon"
    energy_dir.mkdir(parents=True, exist_ok=True)

    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name="moondream_mnist_test",
            output_dir=str(energy_dir),
            output_file="codecarbon_moondream_mnist.csv",
            log_level="error",
            save_to_file=True,
        )

        tracker.start()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = inference_fn(image)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = time.perf_counter() - start_time
        emissions_value = tracker.stop()

        final_data = getattr(
            tracker,
            "final_emissions_data",
            None,
        )

        cpu_energy = float(
            getattr(final_data, "cpu_energy", 0) or 0
        )
        gpu_energy = float(
            getattr(final_data, "gpu_energy", 0) or 0
        )
        ram_energy = float(
            getattr(final_data, "ram_energy", 0) or 0
        )
        total_energy = float(
            getattr(final_data, "energy_consumed", 0)
            or 0
        )

        carbon_intensity = None
        if (
            emissions_value is not None
            and total_energy > 0
        ):
            carbon_intensity = (
                float(emissions_value)
                / total_energy
            )

    else:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = inference_fn(image)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = time.perf_counter() - start_time

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
        gpu_metrics,
    )


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
    input_tokens = int(result.get("input_tokens", 0) or 0)
    output_tokens = int(result.get("output_tokens", 0) or 0)
    total_tokens = input_tokens + output_tokens

    energy_per_token_kwh = 0.0
    joules_per_token = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if total_energy > 0 and total_tokens > 0:
        energy_per_token_kwh = (
            total_energy / total_tokens
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
            gpu_energy / total_energy * 100
        )
        cpu_energy_pct = (
            cpu_energy / total_energy * 100
        )

    return {
        # --- Identity ---
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": COLLECTION_MODE,

        # --- Sample ---
        "sample_index": sample_index,
        "true_label": true_label,
        "prediction": prediction,
        "correct": bool(correct),

        # --- Model identity ---
        "model_type": MODEL_NAME,
        "parameters": MODEL_PARAMETERS,
        "model_flops": MODEL_FLOPS,

        # --- Prediction quality ---
        "confidence_score": (
            round(
                float(result["confidence_score"]),
                6,
            )
            if result.get("confidence_score") is not None
            else None
        ),
        "logit_margin": (
            round(
                float(result["logit_margin"]),
                6,
            )
            if result.get("logit_margin") is not None
            else None
        ),
        "entropy": (
            round(
                float(result["entropy"]),
                6,
            )
            if result.get("entropy") is not None
            else None
        ),

        # --- Timing ---
        "execution_time_sec": round(
            exec_time,
            10,
        ),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh": cpu_energy,
        "gpu_energy_kwh": gpu_energy,
        "ram_energy_kwh": ram_energy,
        "total_energy_kwh": total_energy,
        "total_emissions_kg": emissions_value,
        "carbon_intensity_kgco2_kwh": (
            carbon_intensity
        ),
        "codecarbon_version": CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": (
            round(total_tokens / exec_time, 4)
            if exec_time > 0
            else None
        ),
        "joules_per_token": round(
            joules_per_token,
            8,
        ),
        "energy_per_token_kwh": round(
            energy_per_token_kwh,
            12,
        ),
        "watts_estimated": round(
            watts_estimated,
            4,
        ),
        "gpu_energy_pct_of_total": round(
            gpu_energy_pct,
            2,
        ),
        "cpu_energy_pct_of_total": round(
            cpu_energy_pct,
            2,
        ),

        # --- CPU hardware ---
        "cpu_model": CPU_MODEL_NAME,
        "cpu_architecture": CPU_ARCH,
        "cpu_core_count": CPU_CORE_COUNT,
        "cpu_thread_count": CPU_THREAD_COUNT,
        "cpu_core": CPU_CORE_COUNT,
        "cpu_thread": CPU_THREAD_COUNT,
        "cpu_tdp_w": CPU_TDP_W,
        "cpu_usage_pct": get_cpu_usage(),
        "cpu_clock_mhz": get_cpu_freq(),
        "cpu_temp_c": get_cpu_temp(),
        "cpu_power_draw_w": get_cpu_power_draw_w(),
        "cpu_cores_used": get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model": get_gpu_name(),
        "gpu_core": GPU_CORE_COUNT,
        "gpu_thread": GPU_THREAD_COUNT,
        "gpu_driver_version": GPU_STATIC[
            "gpu_driver_version"
        ],
        "gpu_compute_capability": GPU_STATIC[
            "gpu_compute_capability"
        ],
        "gpu_power_limit_w": GPU_STATIC[
            "gpu_power_limit_w"
        ],
        "gpu_memory_total_mb": GPU_STATIC[
            "gpu_memory_total_mb"
        ],
        "gpu_power_draw_w": gpu_metrics.get(
            "gpu_power_draw_w"
        ),
        "gpu_utilization_pct": gpu_metrics.get(
            "gpu_utilization_pct"
        ),
        "gpu_temp_c": gpu_metrics.get(
            "gpu_temp_c"
        ),
        "gpu_memory_used_mb": gpu_metrics.get(
            "gpu_memory_used_mb"
        ),
        "gpu_sm_clock_mhz": gpu_metrics.get(
            "gpu_sm_clock_mhz"
        ),
        "gpu_memory_clock_mhz": gpu_metrics.get(
            "gpu_memory_clock_mhz"
        ),
        "cuda_driver_version": CUDA_DRIVER_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device_type": str(DEVICE),

        # --- RAM / memory ---
        "ram_usage_pct": (
            psutil.virtual_memory().percent
        ),
        "memory_footprint_mb": (
            get_memory_footprint_mb()
        ),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name": OS_NAME,
        "os_version": OS_VERSION,
        "os_architecture": OS_ARCHITECTURE,
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,

        # --- Final model metrics; backfilled after run ---
        "model_accuracy": None,
        "model_precision_weighted": None,
        "model_recall_weighted": None,
        "model_f1_weighted": None,

        # Keep useful original diagnostics as additional columns.
        "raw_response": result.get(
            "raw_response",
            "",
        ),
        "quality_top_class": result.get(
            "quality_top_class",
            "",
        ),
        "error": error_message,
    }


results = []
true_labels = []
predicted_labels = []
latencies = []
inference_error_count = 0

print("\n" + "=" * 72)
print("TESTING SAVED MNIST FINE-TUNED MODEL")
print("=" * 72)

for test_index in tqdm(
    range(number_of_samples),
    desc="Testing",
):
    image, true_label_id = test_dataset[test_index]

    true_label_id = int(true_label_id)
    true_label = CLASS_NAMES[true_label_id]

    result = {
        "raw_response": "",
        "prediction": "invalid",
        "confidence_score": None,
        "logit_margin": None,
        "entropy": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "quality_top_class": "",
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
        error_message = repr(error)
        inference_error_count += 1

        # Keep timing/hardware fields valid even when inference fails.
        exec_time = 0.0
        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None
        gpu_metrics = get_gpu_metrics()

    prediction = result.get(
        "prediction",
        "invalid",
    )

    correct = prediction == true_label

    true_labels.append(true_label)
    predicted_labels.append(prediction)
    latencies.append(float(exec_time))

    row = build_row(
        sample_index=int(test_index),
        true_label=true_label,
        prediction=prediction,
        correct=correct,
        result=result,
        exec_time=exec_time,
        cpu_energy=cpu_energy,
        gpu_energy=gpu_energy,
        ram_energy=ram_energy,
        total_energy=total_energy,
        emissions_value=emissions_value,
        carbon_intensity=carbon_intensity,
        gpu_metrics=gpu_metrics,
        error_message=error_message,
    )

    results.append(row)

    print(
        f"\nImage {test_index + 1}/{number_of_samples}"
        f"\nTrue label:  {true_label}"
        f"\nPrediction:  {prediction}"
        f"\nRaw answer:  {result.get('raw_response', '')}"
        f"\nConfidence:  {result.get('confidence_score')}"
        f"\nCorrect:     {correct}"
        f"\nLatency:     {exec_time:.4f} seconds"
    )

    if error_message:
        print("Error:       ", error_message)


# ============================================================
# 14. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision_weighted = precision_score(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES,
    average="weighted",
    zero_division=0,
)

f1_weighted = f1_score(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES,
    average="macro",
    zero_division=0,
)

invalid_count = sum(
    prediction == "invalid"
    for prediction in predicted_labels
)

invalid_rate = invalid_count / number_of_samples

average_latency = float(np.mean(latencies))
median_latency = float(np.median(latencies))
minimum_latency = float(np.min(latencies))
maximum_latency = float(np.max(latencies))

report_dictionary = classification_report(
    true_labels,
    predicted_labels,
    labels=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True,
)

matrix_labels = CLASS_NAMES + ["invalid"]

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=matrix_labels,
)


# Backfill final model metrics into every sample row.
for row in results:
    row["model_accuracy"] = float(accuracy)
    row["model_precision_weighted"] = float(
        precision_weighted
    )
    row["model_recall_weighted"] = float(
        recall_weighted
    )
    row["model_f1_weighted"] = float(
        f1_weighted
    )


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 72)
print("FINAL TEST RESULTS")
print("=" * 72)

print(f"Number of images:      {number_of_samples}")
print(f"Accuracy:              {accuracy:.4f}")
print(f"Weighted precision:    {precision_weighted:.4f}")
print(f"Weighted recall:       {recall_weighted:.4f}")
print(f"Weighted F1:           {f1_weighted:.4f}")
print(f"Macro F1:              {macro_f1:.4f}")
print(f"Invalid predictions:   {invalid_count}")
print(f"Invalid rate:          {invalid_rate:.4f}")
print(f"Inference errors:      {inference_error_count}")
print(f"Average latency:       {average_latency:.4f} seconds")
print(f"Median latency:        {median_latency:.4f} seconds")
print(f"Minimum latency:       {minimum_latency:.4f} seconds")
print(f"Maximum latency:       {maximum_latency:.4f} seconds")

print("\nClassification report:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

print("Confusion-matrix label order:")
print(matrix_labels)

print("\nConfusion matrix:")
print(matrix)


# ============================================================
# 16. SAVE PREDICTIONS WITH ALL REQUESTED COLUMNS
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(results).to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = {
    "transformers_version": transformers.__version__,
    "base_model": MODEL_ID,
    "base_revision": MODEL_REVISION,
    "saved_model_directory": str(
        SAVED_MODEL_DIR.resolve()
    ),
    "model_type": MODEL_NAME,
    "parameters": int(MODEL_PARAMETERS),
    "model_flops": MODEL_FLOPS,
    "number_of_samples": int(number_of_samples),
    "device": str(DEVICE),
    "dtype": str(DTYPE),
    "accuracy": float(accuracy),
    "precision_weighted": float(
        precision_weighted
    ),
    "recall_weighted": float(
        recall_weighted
    ),
    "f1_weighted": float(
        f1_weighted
    ),
    "macro_f1": float(macro_f1),
    "invalid_count": int(invalid_count),
    "invalid_rate": float(invalid_rate),
    "inference_error_count": int(
        inference_error_count
    ),
    "average_latency_seconds": average_latency,
    "median_latency_seconds": median_latency,
    "minimum_latency_seconds": minimum_latency,
    "maximum_latency_seconds": maximum_latency,
    "class_names": CLASS_NAMES,
    "confusion_matrix_labels": matrix_labels,
    "confusion_matrix": matrix.tolist(),
    "classification_report": report_dictionary,
    "missing_checkpoint_keys": missing_keys,
    "unexpected_checkpoint_keys": unexpected_keys,
}

METRICS_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with METRICS_JSON.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=2,
    )

print("\nSaved predictions:")
print(OUTPUT_CSV.resolve())
