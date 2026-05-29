"""Configuration scaffolding for the LLM inference calculator.

A full run configuration is composed of two parts::

    Config = GpuConfig + InferConfig

- ``GpuConfig``   : the hardware the workload runs on (Ascend 910B, H200, B200).
- ``InferConfig`` : how inference is parallelized and what shape of requests it serves
                    (TP / DP / EP, prefill-decode disaggregation, batch size, ISL / OSL).

Configs can be built in code or loaded from YAML via :meth:`Config.from_yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class HardwareType(str, Enum):
    """Supported accelerators. Values are the keys used in YAML."""

    ASCEND_910B = "ascend_910b"
    H200 = "h200"
    B200 = "b200"


# Per-device hardware specs live in YAML (not hard-coded) so new accelerators or
# tuned numbers can be added without touching the code.
HARDWARE_PRESETS_PATH = Path(__file__).resolve().parent.parent / "config" / "hardware_presets.yaml"


@lru_cache(maxsize=None)
def load_hardware_presets(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-device specs keyed by ``HardwareType`` value (e.g. ``"h200"``)."""
    with open(path or HARDWARE_PRESETS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _filter_known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that map to a field of the given dataclass."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass
class GpuConfig:
    """Hardware configuration for a single accelerator device.

    Specs left at ``0`` are filled from ``hardware_presets.yaml`` for the chosen
    ``hardware``, so a run config only needs to set fields it wants to override.
    """

    hardware: HardwareType = HardwareType.H200
    compute_tflops: float = 0.0
    memory_gb: float = 0.0
    memory_bw_gbs: float = 0.0
    interconnect_gbs: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.hardware, str):
            self.hardware = HardwareType(self.hardware.lower())
        preset = load_hardware_presets().get(self.hardware.value, {})
        for key, value in preset.items():
            if not getattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GpuConfig":
        return cls(**_filter_known(cls, data or {}))


@dataclass
class InferConfig:
    """Parallelism strategy and request shape for an inference deployment."""

    # Parallelism degrees.
    tp: int = 1  # tensor parallel
    dp: int = 1  # data parallel
    ep: int = 1  # expert parallel (MoE)
    pp: int = 1  # pipeline parallel

    # Serve prefill and decode on separate instances (PD disaggregation).
    pd_disaggregation: bool = False

    # Request shape.
    batch_size: int = 1
    isl: int = 1024  # input sequence length (prompt tokens)
    osl: int = 1024  # output sequence length (generated tokens)

    def __post_init__(self) -> None:
        for name in ("tp", "dp", "ep", "pp", "batch_size", "isl", "osl"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    @property
    def world_size(self) -> int:
        """Devices spanned by one model replica's parallel group."""
        return self.tp * self.pp * self.ep

    @property
    def total_devices(self) -> int:
        """Total devices across all data-parallel replicas."""
        return self.world_size * self.dp

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferConfig":
        return cls(**_filter_known(cls, data or {}))


@dataclass
class Config:
    """Top-level run configuration: ``GpuConfig`` + ``InferConfig``."""

    gpu: GpuConfig = field(default_factory=GpuConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data or {}
        return cls(
            gpu=GpuConfig.from_dict(data.get("gpu", {})),
            infer=InferConfig.from_dict(data.get("infer", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": {
                "hardware": self.gpu.hardware.value,
                "compute_tflops": self.gpu.compute_tflops,
                "memory_gb": self.gpu.memory_gb,
                "memory_bw_gbs": self.gpu.memory_bw_gbs,
                "interconnect_gbs": self.gpu.interconnect_gbs,
            },
            "infer": {
                "tp": self.infer.tp,
                "dp": self.infer.dp,
                "ep": self.infer.ep,
                "pp": self.infer.pp,
                "pd_disaggregation": self.infer.pd_disaggregation,
                "batch_size": self.infer.batch_size,
                "isl": self.infer.isl,
                "osl": self.infer.osl,
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
