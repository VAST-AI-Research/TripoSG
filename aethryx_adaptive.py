"""AETHRYX Adaptive Engine: hardware-aware generation plans."""
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class HardwareSnapshot:
    ram_gib: float
    vram_gib: float | None
    cpu_count: int
    gpu_name: str | None
    storage_free_gib: float


@dataclass(frozen=True)
class EngineProfile:
    target_vram_gib: int
    dtype: str
    quantization: str
    cpu_offload: bool
    sequential_loading: bool
    shared_memory: bool
    generation_enabled: bool
    basic_edit_mode: bool
    quality_priority: bool


@dataclass(frozen=True)
class AdaptivePlan:
    profile: EngineProfile
    hardware: HardwareSnapshot
    reason: str
    oom_recovery: tuple


@dataclass(frozen=True)
class AdaptiveBenchmark:
    engine_id: str
    profile_vram_gib: int
    precision: str
    seconds: float
    peak_vram_mib: float
    faces: int
    vertices: int
    status: str


def detect_hardware():
    ram_gib = 0.0
    try:
        import psutil
        ram_gib = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        ram_gib = 0.0

    storage_free_gib = 0.0
    try:
        usage = shutil.disk_usage(str(Path.cwd()))
        storage_free_gib = usage.free / (1024 ** 3)
    except OSError:
        pass

    gpu_name = None
    vram_gib = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
            if output:
                name, total = output.splitlines()[0].rsplit(",", 1)
                gpu_name = name.strip()
                vram_gib = float(total.strip()) / 1024.0
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    return HardwareSnapshot(
        ram_gib=round(ram_gib, 2),
        vram_gib=round(vram_gib, 2) if vram_gib is not None else None,
        cpu_count=os.cpu_count() or 1,
        gpu_name=gpu_name,
        storage_free_gib=round(storage_free_gib, 2),
    )


TARGETS = (4, 6, 8, 12, 16, 24)


def closest_profile_target(vram_gib):
    if vram_gib is None:
        return 4
    for target in TARGETS:
        if vram_gib <= target + 0.25:
            return target
    return 24


def build_adaptive_plan(hardware=None, engine_id="triposg"):
    hardware = hardware or detect_hardware()
    target = closest_profile_target(hardware.vram_gib)
    basic_edit = hardware.ram_gib > 0 and hardware.ram_gib < 8

    quantization = "none"
    if engine_id != "triposg":
        quantization = "engine-dependent"

    profile = EngineProfile(
        target_vram_gib=target,
        dtype="fp16" if hardware.vram_gib is not None else "fp32",
        quantization=quantization,
        cpu_offload=False if engine_id == "triposg" else target <= 12,
        sequential_loading=False if engine_id == "triposg" else target <= 8,
        shared_memory=True,
        generation_enabled=not basic_edit,
        basic_edit_mode=basic_edit,
        quality_priority=True,
    )

    if hardware.vram_gib is None:
        reason = "Sin GPU NVIDIA detectada; se conserva CPU o modo edición básica."
    elif basic_edit:
        reason = "RAM inferior a 8 GiB: perfil básico de edición."
    else:
        reason = f"Perfil más cercano al VRAM detectado: {target} GiB."

    return AdaptivePlan(
        profile=profile,
        hardware=hardware,
        reason=reason,
        oom_recovery=(
            "clear-cuda-cache",
            "enable-sequential-cpu-offload",
            "retry-once",
        ),
    )


def adaptive_plan_dict(plan):
    return {
        "profile": asdict(plan.profile),
        "hardware": asdict(plan.hardware),
        "reason": plan.reason,
        "oom_recovery": list(plan.oom_recovery),
    }


def benchmark_dict(result):
    return asdict(result)


def clear_cuda_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def unload_models(*objects):
    for obj in objects:
        if obj is not None:
            try:
                del obj
            except Exception:
                pass
    clear_cuda_memory()
