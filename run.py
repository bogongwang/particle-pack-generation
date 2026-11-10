import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import repeat
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.io import load_npy_compressed
from src.simulation_pipeline import add_noise, to_volume_parallel
from src.simulator import NewtonSimulator
from src.workflow import SimulationConfig, simulate_particle_pack


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "GOH_6_250"
DEFAULT_SIM_DATA_DIR = DEFAULT_DATA_ROOT / "sim_data_uint8"
DEFAULT_MANUAL_CHECK_JSON = DEFAULT_DATA_ROOT / "manual_check.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "outputs"

SIZE_GROUPS = ("small", "medium", "large")
DEFAULT_SIZE_PERCENTAGES = {
    "small": 33.0,
    "medium": 33.0,
    "large": 34.0,
}


@dataclass(frozen=True)
class RunConfig:
    sim_data_dir: Path
    manual_check_json: Path
    output_dir: Path
    size_percentages: dict[str, float]
    seed: int
    n_objects: int = 500
    n_simulations: int = 10
    device: str = "cuda:0"
    size_workers: int | None = None
    volume_workers: int = 8


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Generate synthetic particle pack simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim_data_dir",
        type=Path,
        default=DEFAULT_SIM_DATA_DIR,
        help="Directory containing numbered .obj and .npy.gz particle files.",
    )
    parser.add_argument(
        "--manual_check_json",
        type=Path,
        default=DEFAULT_MANUAL_CHECK_JSON,
        help="manual_check.json file used to select objects labelled correct.",
    )
    parser.add_argument(
        "--size_percentages",
        type=float,
        nargs=3,
        metavar=("SMALL", "MEDIUM", "LARGE"),
        default=[33, 33, 34],
        help="Percentages for small, medium, and large object selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for object selection, shuffling, simulation placement, and noise.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where this run's simulation outputs will be written.",
    )
    parser.add_argument(
        "--n_objects",
        type=int,
        default=500,
        help="Number of candidate objects to select before each simulation shuffle.",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=5,
        help="Number of particle pack simulations to run.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Newton/Warp device to run simulation on.",
    )
    parser.add_argument(
        "--size_workers",
        type=int,
        default=None,
        help="Worker processes for measuring object sizes. Defaults to ProcessPoolExecutor's choice.",
    )
    parser.add_argument(
        "--volume_workers",
        type=int,
        default=8,
        help="Worker processes for volume generation.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    size_percentages = dict(zip(SIZE_GROUPS, args.size_percentages, strict=True))
    try:
        validate_percentages(size_percentages)
        validate_positive_int("n-objects", args.n_objects)
        validate_positive_int("n-simulations", args.n_simulations)
        validate_positive_int("volume-workers", args.volume_workers)
        if args.size_workers is not None:
            validate_positive_int("size-workers", args.size_workers)
    except ValueError as exc:
        parser.error(str(exc))

    return RunConfig(
        sim_data_dir=args.sim_data_dir,
        manual_check_json=args.manual_check_json,
        output_dir=output_dir,
        size_percentages=size_percentages,
        seed=args.seed,
        n_objects=args.n_objects,
        n_simulations=args.n_simulations,
        device=args.device,
        size_workers=args.size_workers,
        volume_workers=args.volume_workers,
    )


def validate_percentages(size_percentages: dict[str, float]) -> None:
    for group, percentage in size_percentages.items():
        if percentage < 0:
            raise ValueError(f"{group} percentage must be non-negative, got {percentage}.")

    total = sum(size_percentages.values())
    if not np.isclose(total, 100.0):
        raise ValueError(f"Size percentages must sum to 100, got {total}.")


def validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def load_correct_ids(manual_check_json: Path) -> list[int]:
    with manual_check_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    correct_ids = []
    for item in data["labels"]:
        if item.get("label") != "correct":
            continue

        file_name = item["file_name"]
        correct_ids.append(int(Path(Path(file_name).stem).stem))

    return sorted(correct_ids)


def get_size_stats(obj_id: int, sim_data_dir: Path) -> tuple[int, int]:
    obj = load_npy_compressed(sim_data_dir / f"{obj_id}.npy.gz")
    return obj_id, int(np.sum(obj > 0))


def get_size_group(size: int) -> str:
    if 1e4 <= size < 5e5:
        return "small"
    if 5e5 <= size <= 5e6:
        return "medium"
    if size > 5e6:
        return "large"
    return "unknown"


def measure_object_sizes(
    obj_ids: list[int],
    sim_data_dir: Path,
    workers: int | None,
) -> dict[int, int]:
    sizes = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(get_size_stats, obj_ids, repeat(sim_data_dir))
        for obj_id, size in tqdm(iterator, total=len(obj_ids), desc="Measuring object sizes"):
            sizes[obj_id] = size

    return sizes


def target_counts(n: int, percentages: dict[str, float]) -> dict[str, int]:
    counts = {}
    assigned = 0
    for group in SIZE_GROUPS[:-1]:
        counts[group] = round(n * percentages[group] / 100.0)
        assigned += counts[group]

    counts[SIZE_GROUPS[-1]] = n - assigned
    return counts


def select_ids_by_size(
    id_size_map: dict[int, int],
    n: int,
    percentages: dict[str, float],
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    buckets = {group: [] for group in SIZE_GROUPS}
    unknown_ids = []

    for obj_id, size in id_size_map.items():
        group = get_size_group(size)
        if group in buckets:
            buckets[group].append(obj_id)
        else:
            unknown_ids.append(obj_id)

    if unknown_ids:
        print(f"Skipping {len(unknown_ids)} objects outside size buckets.")

    counts = target_counts(n, percentages)
    selected = []

    for group in SIZE_GROUPS:
        available = buckets[group]
        requested = counts[group]
        if len(available) < requested:
            raise ValueError(
                f"Not enough {group} ids: requested {requested}, available {len(available)}."
            )

        selected.extend(int(obj_id) for obj_id in rng.choice(available, requested, replace=False))

    return selected


def create_simulation_config(seed: int) -> SimulationConfig:
    return SimulationConfig(
        fps=60,
        n_frames=3000,
        sim_substeps=20,
        solver_iters=30,
        obj_scale=0.4,
        obj_spacing=0.5,
        obj_start_z=2,
        container_radius=5,
        container_height=30,
        seed=seed,
    )


def max_simulation_height(sim_res) -> float:
    max_z = 0.0
    for result in sim_res.values():
        max_z = max(max_z, result.translation_zyx[0])
    return max_z


def run_single_simulation(
    sim_idx: int,
    candidate_ids: list[int],
    run_config: RunConfig,
    sim_config: SimulationConfig,
) -> None:
    print(f"{'=' * 50}\nRunning simulation {sim_idx}.\n{'=' * 50}")
    print(f"First 5 shuffled candidate IDs: {candidate_ids[:5]}")

    sim_res = simulate_particle_pack(
        obj_ids=candidate_ids,
        obj_dir=run_config.sim_data_dir,
        sim_config=sim_config,
        device=run_config.device,
    )
    is_simu_correct = NewtonSimulator.verify_simulation_correctness(
        sim_res,
        container_radius=sim_config.container_radius,
    )
    print(f"Is the simulation correct? {is_simu_correct}")

    simulation_result_dir = run_config.output_dir / f"sim_res_{sim_idx}"
    simulation_result_path = simulation_result_dir / "sim_res.json"
    NewtonSimulator.save_simulation_results(
        sim_res,
        simulation_result_path,
        metadata=asdict(sim_config),
    )
    print(f"max_z: {max_simulation_height(sim_res)}")

    if not is_simu_correct:
        return

    metadata = {
        "description": "Synthetic particle pack",
        "simulation_config": asdict(sim_config),
        "run_config": {
            **asdict(run_config),
            "sim_data_dir": str(run_config.sim_data_dir),
            "manual_check_json": str(run_config.manual_check_json),
            "output_dir": str(run_config.output_dir),
        },
        "simulation_path": str(simulation_result_path),
        "timestamp": datetime.now().isoformat(),
    }
    to_volume_parallel(
        simulation_results=sim_res,
        obj_ids=list(sim_res.keys()),
        obj_dir=run_config.sim_data_dir,
        output_dir=simulation_result_dir,
        output_shape=(1024, 1024, 1024),
        obj_scale=sim_config.obj_scale,
        parallel=run_config.volume_workers,
        metadata=metadata,
    )
    add_noise(
        input_path=str(simulation_result_dir / "tomo.zarr"),
        output_path=str(simulation_result_dir / "tomo_noisy.zarr"),
        noise_std=15,
        seed=run_config.seed + sim_idx,
    )


def main() -> None:
    run_config = parse_args()
    sim_config = create_simulation_config(run_config.seed)
    rng = np.random.default_rng(seed=run_config.seed)

    correct_ids = load_correct_ids(run_config.manual_check_json)
    print(f"No. Sim data IDs: {len(correct_ids)}")

    sizes = measure_object_sizes(
        obj_ids=correct_ids,
        sim_data_dir=run_config.sim_data_dir,
        workers=run_config.size_workers,
    )
    candidate_ids = select_ids_by_size(
        id_size_map=sizes,
        n=run_config.n_objects,
        percentages=run_config.size_percentages,
        seed=run_config.seed,
    )
    print(f"First 5 candidate IDs: {candidate_ids[:5]}, total: {len(candidate_ids)}")
    print(f"Writing outputs to: {run_config.output_dir}")

    os.makedirs(run_config.output_dir, exist_ok=True)
    for sim_idx in range(run_config.n_simulations):
        shuffled_ids = list(candidate_ids)
        rng.shuffle(shuffled_ids)
        run_single_simulation(
            sim_idx=sim_idx,
            candidate_ids=shuffled_ids,
            run_config=run_config,
            sim_config=sim_config,
        )


if __name__ == "__main__":
    main()
