"""Demo: load a SuperCalc config from YAML and print the parsed result.

运行方式 / How to run (from the repo root):

    python examples/compute_demo.py
    python examples/compute_demo.py examples/example_config.yaml   # custom config

The script loads the config, fills hardware specs from presets, and prints the
deployment shape. This is where the actual inference cost calculation would plug in.
"""

import sys
from pathlib import Path

# Make the repo root importable so `from super_calc import Config` works when this
# script is run directly (python examples/compute_demo.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from super_calc import Config  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "example_config.yaml"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = Config.from_yaml(config_path)

    gpu, infer = config.gpu, config.infer
    print(f"Loaded config from: {config_path}\n")
    print("Hardware")
    print(f"  device           : {gpu.hardware.value}")
    print(f"  compute (TFLOPS) : {gpu.compute_tflops}")
    print(f"  memory (GB)      : {gpu.memory_gb}")
    print(f"  mem BW (GB/s)    : {gpu.memory_bw_gbs}")
    print(f"  interconnect     : {gpu.interconnect_gbs} GB/s")
    print("\nInference")
    print(f"  parallelism      : TP={infer.tp} DP={infer.dp} EP={infer.ep} PP={infer.pp}")
    print(f"  PD disaggregation: {infer.pd_disaggregation}")
    print(f"  batch size       : {infer.batch_size}")
    print(f"  ISL / OSL        : {infer.isl} / {infer.osl}")
    print(f"  devices/replica  : {infer.world_size}")
    print(f"  total devices    : {infer.total_devices}")

    # TODO: feed `config` into the inference cost model here.


if __name__ == "__main__":
    main()
