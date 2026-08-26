import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.models import mobilenet_v2
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

NUM_TEST_SAMPLES = 10

#NUM_TEST_SAMPLES = None

MODEL_PATH = "mobilenet_v2_mnist_cpu.pt"
MNIST_ROOT = "./mnist_data"
DEVICE_MODE = "cpu"          # "cpu", "cuda", or "auto"
ENABLE_CODECARBON = True
CLASS_NAMES = [str(i) for i in range(10)]

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Device
# ============================================================
def resolve_device():
    mode = DEVICE_MODE.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = 'cpu'#resolve_device()

# ============================================================
# Optional packages
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

try:
    import cpuinfo
    _CPU_INFO = cpuinfo.get_cpu_info()
    CPU_MODEL_NAME = _CPU_INFO.get("brand_raw", "Unknown")
    CPU_ARCH = _CPU_INFO.get("arch", platform.machine())
except Exception:
    CPU_MODEL_NAME = platform.processor() or "Unknown"
    CPU_ARCH = platform.machine()

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0) if torch.cuda.is_available() else None
except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False

# ============================================================
# Environment / identity
# ============================================================
TORCH_VERSION = torch.__version__
PYTHON_VERSION = sys.version.split()[0]
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
SYSTEM_RAM_TOTAL_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)
CPU_TDP_W = None


def get_os_full_name():
    return f"{platform.system()} {platform.release()} {platform.version()} {platform.machine()}"

OS_FULL_NAME = get_os_full_name()


def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]
DEVICE_LOG_DIR = OUTPUT_ROOT / DEVICE_SHORT
DEVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DEVICE_LOG_DIR / "mobilenet_v2_mnist_test_full_telemetry.csv"
#SUMMARY_JSON = DEVICE_LOG_DIR / "mobilenet_v2_mnist_test_summary.json"
CODECARBON_CSV_PATH = OUTPUT_ROOT / "codecarbon" / "mobilenet_codecarbon_mnist.csv"
os.makedirs(CODECARBON_CSV_PATH, exist_ok=True)
# ============================================================
# Hardware helpers
# ============================================================
def get_hostname():
    return socket.gethostname()


def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_cpu_freq():
    try:
        f = psutil.cpu_freq()
        return round(f.current, 2) if f else None
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                vals = [x.current for x in temps[key] if x.current and x.current > 0]
                if vals:
                    return round(sum(vals) / len(vals), 1)
    except Exception:
        pass
    return None


def get_cpu_power_draw_w():
    return None


def get_cpu_cores_used():
    try:
        return sum(1 for x in psutil.cpu_percent(percpu=True) if x > 1.0)
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 4)
    except Exception:
        return None


def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"


def get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        x = pynvml.nvmlSystemGetDriverVersion()
        return x.decode() if isinstance(x, bytes) else x
    except Exception:
        return None

CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static():
    out = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
    }
    if not torch.cuda.is_available() or not NVML_AVAILABLE or NVML_HANDLE is None:
        return out
    try:
        props = torch.cuda.get_device_properties(0)
        out["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        mem = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        out["gpu_memory_total_mb"] = round(mem.total / (1024 ** 2), 2)
        out["gpu_power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(NVML_HANDLE) / 1000.0, 2)
    except Exception:
        pass
    return out

GPU_STATIC = get_gpu_static()


def get_gpu_core_thread():
    if not torch.cuda.is_available():
        return None, None
    try:
        p = torch.cuda.get_device_properties(0)
        cores_per_sm = {2: 32, 3: 192, 5: 128, 6: 64, 7: 64, 8: 128, 9: 128}.get(p.major, 64)
        return p.multi_processor_count * cores_per_sm, p.multi_processor_count * p.max_threads_per_multi_processor
    except Exception:
        return None, None

GPU_CORE_COUNT, GPU_THREAD_COUNT = get_gpu_core_thread()


def get_gpu_metrics():
    out = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
    }
    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return out
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        mem = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        out.update({
            "gpu_power_draw_w": round(pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000.0, 2),
            "gpu_utilization_pct": util.gpu,
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_memory_used_mb": round(mem.used / (1024 ** 2), 2),
            "gpu_sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE, pynvml.NVML_CLOCK_SM),
            "gpu_memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE, pynvml.NVML_CLOCK_MEM),
        })
    except Exception:
        pass
    return out

# ============================================================
# Model helpers
# ============================================================
def get_prediction_quality(logits):
    probs = F.softmax(logits, dim=1).squeeze(0)
    confidence = float(probs.max().item())
    top2 = torch.topk(logits.squeeze(0), k=2).values
    logit_margin = float((top2[0] - top2[1]).item())
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return round(confidence, 6), round(logit_margin, 6), round(entropy, 6)


def compute_model_flops(model):
    if not FVCORE_AVAILABLE:
        return None
    try:
        dummy = torch.ones((1, 3, 224, 224), device=DEVICE)
        a = FlopCountAnalysis(model, dummy)
        a.unsupported_ops_warnings(False)
        a.uncalled_modules_warnings(False)
        return int(a.total())
    except Exception as e:
        print(f"FLOPs unavailable: {e}")
        return None

# ============================================================
# Energy-tracked inference
# ============================================================
def run_model_inference(model, image):
    tracker = None
    emissions_value = 0.0
    cpu_energy = gpu_energy = ram_energy = total_energy = 0.0
    carbon_intensity = None

    if ENABLE_CODECARBON and CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name="mobilenet_v2_mnist_edge_inference",
            output_dir=CODECARBON_CSV_PATH,
            output_file=CODECARBON_CSV_PATH.name,
            log_level="error",
            save_to_file=True,
            measure_power_secs=1,
        )
        tracker.start()

    # if DEVICE.type == "cuda":
    #     torch.cuda.synchronize()
    
    with torch.inference_mode():
        logits = model(image)
    # if DEVICE.type == "cuda":
    #     torch.cuda.synchronize()

    t0 = time.perf_counter()
    exec_time = time.perf_counter() - t0

    if tracker is not None:
        emissions_value = float(tracker.stop() or 0.0)
        fd = getattr(tracker, "final_emissions_data", None)
        if fd is not None:
            cpu_energy = float(getattr(fd, "cpu_energy", 0) or 0)
            gpu_energy = float(getattr(fd, "gpu_energy", 0) or 0)
            ram_energy = float(getattr(fd, "ram_energy", 0) or 0)
            total_energy = float(getattr(fd, "energy_consumed", 0) or 0)
        if total_energy > 0:
            carbon_intensity = emissions_value / total_energy

    prediction = int(torch.argmax(logits, dim=1).item())
    return (
        prediction, logits, exec_time,
        cpu_energy, gpu_energy, ram_energy, total_energy,
        emissions_value, carbon_intensity, get_gpu_metrics()
    )

# ============================================================
# Row builder
# ============================================================
def build_row(sample_index, true_label, prediction, logits, exec_time,
              model_name, parameters, model_flops,
              cpu_energy, gpu_energy, ram_energy, total_energy,
              emissions_value, carbon_intensity, gpu_metrics):

    confidence_score, logit_margin, entropy = get_prediction_quality(logits)
    correct = int(prediction) == int(true_label)

    # Pixel/value proxy for MobileNetV2 input tensor [3,224,224].
    input_tokens = 3 * 224 * 224
    output_tokens = 10
    total_tokens = input_tokens + output_tokens

    total_energy = float(total_energy or 0.0)
    cpu_energy = float(cpu_energy or 0.0)
    gpu_energy = float(gpu_energy or 0.0)
    ram_energy = float(ram_energy or 0.0)

    energy_per_token_kwh = total_energy / total_tokens if total_energy > 0 else 0.0
    joules_per_token = total_energy * 3_600_000 / total_tokens if total_energy > 0 else 0.0
    watts_estimated = total_energy * 3_600_000 / exec_time if total_energy > 0 and exec_time > 0 else 0.0
    gpu_energy_pct = gpu_energy / total_energy * 100 if total_energy > 0 else 0.0
    cpu_energy_pct = cpu_energy / total_energy * 100 if total_energy > 0 else 0.0

    return {
        # --- Identity ---
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": "automated_edge",

        # --- Sample ---
        "sample_index": sample_index,
        "true_label": int(true_label),
        "prediction": int(prediction),
        "correct": correct,

        # --- Model identity ---
        "model_type": model_name,
        "parameters": parameters,
        "model_flops": model_flops,

        # --- Prediction quality ---
        "confidence_score": confidence_score,
        "logit_margin": logit_margin,
        "entropy": entropy,

        # --- Timing ---
        "execution_time_sec": round(exec_time, 10),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh": cpu_energy,
        "gpu_energy_kwh": gpu_energy,
        "ram_energy_kwh": ram_energy,
        "total_energy_kwh": total_energy,
        "total_emissions_kg": emissions_value,
        "carbon_intensity_kgco2_kwh": carbon_intensity,
        "codecarbon_version": CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / exec_time, 4) if exec_time > 0 else None,
        "joules_per_token": round(joules_per_token, 8),
        "energy_per_token_kwh": round(energy_per_token_kwh, 12),
        "watts_estimated": round(watts_estimated, 8),
        "gpu_energy_pct_of_total": round(gpu_energy_pct, 4),
        "cpu_energy_pct_of_total": round(cpu_energy_pct, 4),

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
        "gpu_driver_version": GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability": GPU_STATIC["gpu_compute_capability"],
        "gpu_power_limit_w": GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb": GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w": gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct": gpu_metrics.get("gpu_utilization_pct"),
        "gpu_temp_c": gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb": gpu_metrics.get("gpu_memory_used_mb"),
        "gpu_sm_clock_mhz": gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz": gpu_metrics.get("gpu_memory_clock_mhz"),
        "cuda_driver_version": CUDA_DRIVER_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device_type": str(DEVICE),

        # --- RAM / memory ---
        "ram_usage_pct": psutil.virtual_memory().percent,
        "memory_footprint_mb": get_memory_footprint_mb(),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name": OS_NAME,
        "os_version": OS_VERSION,
        "os_architecture": OS_ARCHITECTURE,
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,

        # --- Final model metrics (backfilled after run) ---
        "model_accuracy": None,
        "model_precision_weighted": None,
        "model_recall_weighted": None,
        "model_f1_weighted": None,
    }

# ============================================================
# MNIST preprocessing - MUST match training
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# ============================================================
# Load test dataset
# ============================================================
test_dataset = torchvision.datasets.MNIST(
    root=MNIST_ROOT,
    train=False,
    download=True,
    transform=transform,
)

number_of_samples = len(test_dataset) if NUM_TEST_SAMPLES is None else min(NUM_TEST_SAMPLES, len(test_dataset))

# ============================================================
# Rebuild and load MobileNetV2
# ============================================================
model = mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 10)

try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
except TypeError:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE)
model.eval()

MODEL_NAME = "MobileNetV2_MNIST"
PARAMETERS = int(sum(p.numel() for p in model.parameters()))
MODEL_FLOPS = compute_model_flops(model)


# ============================================================
# Test one image at a time
# ============================================================
rows = []
true_labels = []
predictions = []

print("=" * 70)
print("MOBILENETV2 MNIST TEST WITH FULL TELEMETRY")
print("=" * 70)
print("Device:", DEVICE)
print("Samples:", number_of_samples)
print("Parameters:", PARAMETERS)
print("FLOPs:", MODEL_FLOPS)

for sample_index in tqdm(range(number_of_samples), desc="Testing"):
    image, true_label = test_dataset[sample_index]
    image = image.unsqueeze(0).to(DEVICE)

    (prediction, logits, exec_time,
     cpu_energy, gpu_energy, ram_energy, total_energy,
     emissions_value, carbon_intensity, gpu_metrics) = run_model_inference(model, image)

    rows.append(build_row(
        sample_index=sample_index,
        true_label=true_label,
        prediction=prediction,
        logits=logits,
        exec_time=exec_time,
        model_name=MODEL_NAME,
        parameters=PARAMETERS,
        model_flops=MODEL_FLOPS,
        cpu_energy=cpu_energy,
        gpu_energy=gpu_energy,
        ram_energy=ram_energy,
        total_energy=total_energy,
        emissions_value=emissions_value,
        carbon_intensity=carbon_intensity,
        gpu_metrics=gpu_metrics,
    ))

    true_labels.append(int(true_label))
    predictions.append(int(prediction))

# ============================================================
# Final metrics + backfill
# ============================================================
accuracy = float(accuracy_score(true_labels, predictions))
report = classification_report(
    true_labels,
    predictions,
    labels=list(range(10)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
precision_weighted = float(report["weighted avg"]["precision"])
recall_weighted = float(report["weighted avg"]["recall"])
f1_weighted = float(report["weighted avg"]["f1-score"])
macro_f1 = float(report["macro avg"]["f1-score"])
confusion = confusion_matrix(true_labels, predictions, labels=list(range(10)))

for row in rows:
    row["model_accuracy"] = accuracy
    row["model_precision_weighted"] = precision_weighted
    row["model_recall_weighted"] = recall_weighted
    row["model_f1_weighted"] = f1_weighted

# ============================================================
# Save telemetry CSV
# ============================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# summary = {
#     "model_type": MODEL_NAME,
#     "dataset": "MNIST",
#     "tested_samples": number_of_samples,
#     "device": str(DEVICE),
#     "parameters": PARAMETERS,
#     "model_flops": MODEL_FLOPS,
#     "accuracy": accuracy,
#     "precision_weighted": precision_weighted,
#     "recall_weighted": recall_weighted,
#     "f1_weighted": f1_weighted,
#     "macro_f1": macro_f1,
#     "confusion_matrix": confusion.tolist(),
# }
# SUMMARY_JSON.write_text(json.dumps(summary, indent=4), encoding="utf-8")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Accuracy:           {accuracy:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1:        {f1_weighted:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print("\nConfusion matrix:")
#print(confusion)
print("\nTelemetry CSV:", OUTPUT_CSV)
#print("Summary JSON:", SUMMARY_JSON)
