import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import newton
from tqdm import tqdm
import trimesh
from pptools.io import ZarrWriter
from scipy.ndimage import affine_transform, zoom
from scipy.spatial.transform import Rotation

from .simulator import NewtonSimulator, SimulationResult
from .io import load_npy_compressed


@dataclass
class SimulationConfig:
    DEFAULT_SCALE: float = 0.01
    DEFAULT_WALL_THICKNESS: float = 0.25

    fps: int = 60
    n_frames: int = 1800
    sim_substeps: int = 20
    solver_iters: int = 30

    obj_scale: float = 0.5
    obj_spacing: float = 0.5
    obj_start_z: float = 2.0

    container_radius: float = 5.0
    container_height: float = 10.0

    seed: int | None = None

    @property
    def sim_scale(self):
        return self.DEFAULT_SCALE * self.obj_scale


def simulate_particle_pack(
    obj_ids: list[int],
    obj_dir: str | Path,
    sim_config: SimulationConfig | None = None,
    device: str = 'cuda:0',
    viewer: newton.viewer.ViewerBase | None = None,
):
    if sim_config is None:
        sim_config = SimulationConfig()

    simulator = NewtonSimulator(
        device=device,
        viewer=viewer
    )

    simulator.add_cylinder(
        radius=sim_config.container_radius,
        height=sim_config.container_height,
        wall_thickness=sim_config.DEFAULT_WALL_THICKNESS,
        floor_thickness=sim_config.DEFAULT_WALL_THICKNESS,
        n_faces=64,
        center=(0.0, 0.0, 0)
    )

    rng = np.random.default_rng(seed=sim_config.seed)

    curr_z = sim_config.obj_start_z
    for id in tqdm(obj_ids, desc="Adding particles to simulator"):
        mesh_file = trimesh.load(os.path.join(obj_dir, f'{id}.obj'), force='mesh')
        prepared_mesh = simulator._prepare_mesh(mesh_file, scale=sim_config.sim_scale, center_mesh=True)

        mesh_z, mesh_y, mesh_x = prepared_mesh.bounds[1, :] - prepared_mesh.bounds[0, :]
        curr_z += mesh_z * 0.5 + sim_config.obj_spacing
        rand_x = rng.uniform(
            -sim_config.container_radius + mesh_x * 0.5,
            sim_config.container_radius - mesh_x * 0.5
        ) * 0.7
        rand_y = rng.uniform(
            -sim_config.container_radius + mesh_y * 0.5,
            sim_config.container_radius - mesh_y * 0.5
        ) * 0.7

        simulator.add_particle(
            id=id,
            mesh=prepared_mesh,
            position=(rand_x, rand_y, curr_z),
            build_sdf=True,
        )

        curr_z += mesh_z * 0.5 + sim_config.obj_spacing

    simulator.finalize(
        solver="XPBD",
        solver_iterations=sim_config.solver_iters,
    )

    simulator.simulate(
        fps=sim_config.fps,
        sim_substeps=sim_config.sim_substeps,
        num_frames=sim_config.n_frames,
    )

    simulation_results = simulator.get_simulation_results()
    del simulator
    return simulation_results
