import os
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import newton
from tqdm import tqdm
import trimesh
from pptools.io import ZarrWriter, ZarrReader
from scipy.ndimage import affine_transform, zoom
from scipy.spatial.transform import Rotation
import dask.array as da

from .simulator import NewtonSimulator, SimulationResult
from .io import load_npy_compressed


DEFAULT_SCALE = 0.01


@dataclass
class SimulationConfig:
    fps: int = 60
    n_frames: int = 100
    sim_substeps: int = 10
    solver_iters: int = 30

    obj_scale: float = 1.0
    obj_spacing: float = 0.5
    obj_start_z: float = 2.0

    container_radius: float = 5.0
    container_height: float = 10.0

    seed: int = 42

def simulate_box(
    obj_ids: list[int],
    obj_dir: str,
    cylinder_radius: float = 5.0,
    cylinder_height: float = 10.0,
    obj_scale: float = 1.0,
    obj_spacing: float = 0.5,
    frames: int = 600,
    sim_substeps: int = 20,
    device: str = "cuda:0",
    viewer: newton.viewer.ViewerBase | None = None,
) -> dict[int, SimulationResult]:
    sim_scale = obj_scale * DEFAULT_SCALE

    box_simulator = NewtonSimulator(
        device=device,
        viewer=viewer
    )

    box_simulator.add_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        wall_thickness=0.25,
        floor_thickness=0.25,
        n_faces=64,
        center=(0.0, 0.0, 0.0)
    )

    curr_z = 0
    for id in obj_ids:
        np_file = load_npy_compressed(os.path.join(obj_dir, f'{id}.npy.gz'))
        shape = np.array(np_file.shape)[::-1]
        scaled_shape = shape * sim_scale
        curr_z += np.max(scaled_shape) * 0.5 + obj_spacing
        rand_xy = np.random.uniform(-cylinder_radius * 0.5, cylinder_radius * 0.5, size=2)
        box_simulator.add_cuboid(
            id=id,
            half_extents=scaled_shape * 0.5,
            position=(rand_xy[0], rand_xy[1], curr_z),
        )
        curr_z += np.max(scaled_shape) * 0.5 + obj_spacing

    box_simulator.finalize(
        solver="XPBD",
        solver_iterations=30,
    )

    box_simulator.simulate(
        fps=60,
        sim_substeps=sim_substeps,
        num_frames=frames,
    )

    return box_simulator.get_simulation_results()


def simulate_convex(
    obj_ids: list[int],
    obj_dir: str,
    simulation_results: list[SimulationResult] | None = None,
    cylinder_radius: float = 5.0,
    cylinder_height: float = 10.0,
    obj_scale: float = 1.0,
    obj_spacing: float = 0.5,
    obj_start_z: float = 0.0,
    frames: int = 600,
    sim_substeps: int = 20,
    seed: int = 42,
    device: str = "cuda:0",
    viewer: newton.viewer.ViewerBase | None = None,
) -> dict[int, SimulationResult]:
    convex_simulator = NewtonSimulator(
        device=device,
        viewer=viewer
    )

    convex_simulator.add_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        wall_thickness=0.25,
        floor_thickness=0.25,
        n_faces=64,
        center=(0.0, 0.0, 0.0)
    )

    rng = np.random.default_rng(seed=seed)

    sim_scale = obj_scale * DEFAULT_SCALE
    if simulation_results is None:
        curr_z = obj_start_z
        for id in tqdm(obj_ids, desc="Adding particles to simulator"):
            mesh_file = trimesh.load(os.path.join(obj_dir, f'{id}.obj'), force='mesh')
            prepared_mesh = convex_simulator._prepare_mesh(mesh_file, scale=sim_scale, center_mesh=True)
            mesh_z, mesh_y, mesh_x = prepared_mesh.bounds[1, :] - prepared_mesh.bounds[0, :]
            curr_z += mesh_z * 0.5 + obj_spacing
            rand_x = rng.uniform(-cylinder_radius + mesh_x * 0.5, cylinder_radius - mesh_x * 0.5) * 0.7
            rand_y = rng.uniform(-cylinder_radius + mesh_y * 0.5, cylinder_radius - mesh_y * 0.5) * 0.7
            convex_simulator.add_particle(
                id=id,
                mesh=prepared_mesh,
                position=(rand_x, rand_y, curr_z),
            )
            curr_z += mesh_z * 0.5 + obj_spacing
    else:
        for id in tqdm(obj_ids, desc="Adding particles to simulator"):
            mesh_file = trimesh.load(os.path.join(obj_dir, f'{id}.obj'), force='mesh')
            prepared_mesh = convex_simulator._prepare_mesh(mesh_file, scale=sim_scale, center_mesh=True)
            if id not in simulation_results:
                warnings.warn(f"Simulation results for object ID {id} not found.")
            translation = simulation_results[id].translation_xyz
            rotation = simulation_results[id].rotation_xyzw
            convex_simulator.add_particle(
                id=id,
                mesh=prepared_mesh,
                position=translation,
                quat_xyzw=rotation,
            )

    convex_simulator.finalize(
        solver="XPBD",
        solver_iterations=30,
    )

    convex_simulator.simulate(
        fps=60,
        sim_substeps=sim_substeps,
        num_frames=frames,
    )

    return convex_simulator.get_simulation_results()


def to_volume(
    simulation_results: dict[int, SimulationResult],
    obj_ids: list[int],
    obj_dir: str,
    output_dir: str,
    output_shape: tuple[int, int, int] | None = None,
    cylinder_radius: float = 5.0,
    cylinder_height: float = 10.0,
    obj_scale: float = 1.0,
    metadata: dict | None = None,
):
    if output_shape is None:
        output_shape = (
            int(cylinder_height / DEFAULT_SCALE),
            int(cylinder_radius * 2 / DEFAULT_SCALE),
            int(cylinder_radius * 2 / DEFAULT_SCALE),
        )

    output_tomo = ZarrWriter(
        path=os.path.join(output_dir, 'tomo.zarr'),
        shape=output_shape,
        dtype=np.uint8,
        chunks=(128, 128, 128),
        comp_level=0,
        overwrite=True,
        attributes=metadata,
    )
    output_mask = ZarrWriter(
        path=os.path.join(output_dir, 'mask.zarr'),
        shape=output_shape,
        dtype=np.uint8,
        chunks=(128, 128, 128),
        comp_level=1,
        overwrite=True,
        attributes=metadata,
    )

    for id in tqdm(obj_ids):
        if id not in simulation_results:
            warnings.warn(f"Simulation results for object ID {id} not found. Skipping.")
            continue

        sim_result = simulation_results[id]
        volume = load_npy_compressed(os.path.join(obj_dir, f'{id}.npy.gz'))

        translation = np.array(sim_result.translation_zyx) / DEFAULT_SCALE
        rotation = sim_result.rotation_xyzw

        if translation[0] < 0:
            print(f"Object ID {id} has negative z-translation {translation[0]:.2f}. Skipping.")
            continue

        resized_volume = resize_volume(
            volume=volume,
            scale_factor=obj_scale,
            order=1
        )
        rotated_volume = rotate_quat(
            volume=resized_volume,
            q_xyzw=rotation,
            order=1
        )
        unsafe_offset = np.round(
            (np.array([0, output_shape[1], output_shape[2]]) / 2)
            + translation
            - (np.array(rotated_volume.shape) / 2)
        ).astype(int)
        write_offset, crop_before_write = crop_box_to_ref(
            obj_start=unsafe_offset,
            obj_shape=rotated_volume.shape,
            ref_shape=output_shape,
        )
        if write_offset is None:
            print(f"Object ID {id} is completely outside the output volume. Skipping.")
            continue

        rotated_volume = rotated_volume[crop_before_write]
        output_tomo.write(
            rotated_volume.astype(np.uint8),
            offset=write_offset,
            write_positive_regions_only=True,
        )
        output_mask.write(
            (rotated_volume > 0).astype(np.int32) * id,
            offset=write_offset,
            write_positive_regions_only=True,
        )



def _prepare_volume_item(args):
    (
        id,
        sim_result,
        obj_dir,
        output_shape,
        obj_scale,
    ) = args

    volume = load_npy_compressed(os.path.join(obj_dir, f'{id}.npy.gz'))

    translation = np.array(sim_result.translation_zyx) / DEFAULT_SCALE
    rotation = sim_result.rotation_xyzw

    if translation[0] < 0:
        return id, None, f"Object ID {id} has negative z-translation {translation[0]:.2f}. Skipping."

    resized_volume = resize_volume(
        volume=volume,
        scale_factor=obj_scale,
        order=1,
    )

    rotated_volume = rotate_quat(
        volume=resized_volume,
        q_xyzw=rotation,
        order=1,
    )

    unsafe_offset = np.round(
        (np.array([0, output_shape[1], output_shape[2]]) / 2)
        + translation
        - (np.array(rotated_volume.shape) / 2)
    ).astype(int)

    write_offset, crop_before_write = crop_box_to_ref(
        obj_start=unsafe_offset,
        obj_shape=rotated_volume.shape,
        ref_shape=output_shape,
    )

    if write_offset is None:
        return id, None, f"Object ID {id} is completely outside the output volume. Skipping."

    rotated_volume = rotated_volume[crop_before_write].astype(np.uint8)

    return id, (rotated_volume, write_offset), None


def to_volume_parallel(
    simulation_results: dict[int, SimulationResult],
    obj_ids: list[int],
    obj_dir: str,
    output_dir: str,
    output_shape: tuple[int, int, int] | None = None,
    cylinder_radius: float = 5.0,
    cylinder_height: float = 10.0,
    obj_scale: float = 1.0,
    parallel: int = 1,
    metadata: dict | None = None,
):
    if output_shape is None:
        output_shape = (
            int(cylinder_height / DEFAULT_SCALE),
            int(cylinder_radius * 2 / DEFAULT_SCALE),
            int(cylinder_radius * 2 / DEFAULT_SCALE),
        )

    output_tomo = ZarrWriter(
        path=os.path.join(output_dir, 'tomo.zarr'),
        shape=output_shape,
        dtype=np.uint8,
        chunks=(128, 128, 128),
        comp_level=0,
        overwrite=True,
        attributes=metadata,
    )

    output_mask = ZarrWriter(
        path=os.path.join(output_dir, 'mask.zarr'),
        shape=output_shape,
        dtype=np.int32,
        chunks=(128, 128, 128),
        comp_level=1,
        overwrite=True,
        attributes=metadata,
    )

    jobs = []
    for id in obj_ids:
        if id not in simulation_results:
            warnings.warn(f"Simulation results for object ID {id} not found. Skipping.")
            continue

        jobs.append((
            id,
            simulation_results[id],
            obj_dir,
            output_shape,
            obj_scale,
        ))

    if parallel <= 1:
        iterator = map(_prepare_volume_item, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=parallel)
        iterator = executor.map(_prepare_volume_item, jobs)

    try:
        for id, result, message in tqdm(iterator, total=len(jobs)):
            if message is not None:
                print(message)
                continue

            if result is None:
                continue

            rotated_volume, write_offset = result

            output_tomo.write(
                rotated_volume,
                offset=write_offset,
                write_positive_regions_only=True,
            )

            output_mask.write(
                (rotated_volume > 0).astype(np.int32) * id,
                offset=write_offset,
                write_positive_regions_only=True,
            )
    finally:
        if parallel > 1:
            executor.shutdown()


def resize_volume(
    volume: np.ndarray,
    scale_factor: float = 1.0,
    order: int = 1,
):
    # Use order = 0 for masks/labels and order=1 for tomograms
    if scale_factor == 1.0:
        return volume
    scale = (scale_factor, scale_factor, scale_factor)
    return zoom(volume, zoom=scale, order=order, mode='constant', cval=0.0)


def rotate_quat(
    volume: np.ndarray,
    q_xyzw: tuple[float, float, float, float],
    order: int = 1,
) -> np.ndarray:
    """
    Rotate a zyx volume around its image centre using an xyzw quaternion.

    The output shape is expanded so the rotated particle is not cropped.
    Use order=0 for masks/labels and order=1 for tomograms.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    r_xyz = Rotation.from_quat(q_xyzw).as_matrix()
    xyz_from_zyx = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    r_zyx = xyz_from_zyx.T @ r_xyz @ xyz_from_zyx

    in_shape = np.asarray(volume.shape, dtype=float)
    in_center = (in_shape - 1.0) / 2.0
    corners = np.array(
        [
            [z, y, x]
            for z in (0.0, in_shape[0] - 1.0)
            for y in (0.0, in_shape[1] - 1.0)
            for x in (0.0, in_shape[2] - 1.0)
        ],
        dtype=float,
    )
    rotated_corners = (corners - in_center) @ r_zyx.T
    out_shape = np.ceil(rotated_corners.max(axis=0) - rotated_corners.min(axis=0) + 1.0).astype(int)
    out_center = (out_shape - 1.0) / 2.0

    inverse_matrix = r_zyx.T
    offset = in_center - inverse_matrix @ out_center
    return affine_transform(
        volume,
        matrix=inverse_matrix,
        offset=offset,
        output_shape=tuple(out_shape),
        order=order,
        mode="constant",
    )


def crop_box_to_ref(obj_start, obj_shape, ref_shape):
    """
    Crop a 3D box to fit inside a reference box starting at (0, 0, 0).

    Parameters
    ----------
    box2_start:
        (z, y, x) start coordinate of box2 in reference coordinates.
        Can contain negative values.

    box2_shape:
        (d, h, w) shape of box2.

    ref_shape:
        (D, H, W) shape of reference/container.

    Returns
    -------
    new_start:
        (z, y, x) valid start coordinate in the reference box.

    box2_slices:
        Slices to apply to box2.

    valid_shape:
        Shape after cropping.

    Returns None if there is no overlap.
    """
    z, y, x = obj_start
    d, h, w = obj_shape
    D, H, W = ref_shape

    # Box2 end coordinate in reference coordinates
    z_end = z + d
    y_end = y + h
    x_end = x + w

    # Intersection in reference coordinates
    new_z = max(z, 0)
    new_y = max(y, 0)
    new_x = max(x, 0)

    new_z_end = min(z_end, D)
    new_y_end = min(y_end, H)
    new_x_end = min(x_end, W)

    # Valid cropped shape
    new_d = new_z_end - new_z
    new_h = new_y_end - new_y
    new_w = new_x_end - new_x

    if new_d <= 0 or new_h <= 0 or new_w <= 0:
        print(f'new_z={new_z}, new_y={new_y}, new_x={new_x}')
        print(f'new_z_end={new_z_end}, new_y_end={new_y_end}, new_x_end={new_x_end}')
        if new_d <= 0:
            print(f'new_d={new_d} <= 0')
        if new_h <= 0:
            print(f'new_h={new_h} <= 0')
        if new_w <= 0:
            print(f'new_w={new_w} <= 0')
        # print(f'')
        return None, None

    # Corresponding slicing indices in box2 local coordinates
    src_z0 = new_z - z
    src_y0 = new_y - y
    src_x0 = new_x - x

    src_z1 = src_z0 + new_d
    src_y1 = src_y0 + new_h
    src_x1 = src_x0 + new_w

    box2_slices = (
        slice(src_z0, src_z1),
        slice(src_y0, src_y1),
        slice(src_x0, src_x1),
    )

    new_start = (new_z, new_y, new_x)

    return new_start, box2_slices




def _add_gaussian_noise(
    image: da.Array,
    noise_std=5.0,
    seed=None,
    background_only=False
):
    """
    Add Gaussian noise to a uint8 Dask image lazily.

    Parameters
    ----------
    image : dask.array.Array
        Input image, dtype uint8, values in [0, 255].
    noise_std : float
        Standard deviation of Gaussian noise in uint8 intensity units.
        For example:
        - 1-3: subtle noise
        - 5-10: moderate noise
        - 15-25: strong noise
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    noisy : dask.array.Array
        Noisy image, dtype uint8.
    """
    if image.dtype != np.uint8:
        raise ValueError("Expected a uint8 Dask array")

    rng = da.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=image.shape,
        chunks=image.chunks,
    )
    image = image.astype(np.float32)
    if background_only:
        noisy_image = da.where(image != 0, image, image + noise)
    else:
        noisy_image = image + noise
    noisy_min = -noise_std * 3
    noisy_max = 255 + noise_std * 3
    noisy_image = ((noisy_image - noisy_min) / (noisy_max - noisy_min) * 255)
    noisy_image = da.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_noise(
    input_path: str,
    output_path: str,
    noise_std: float = 5.0,
    background_only: bool = False,
    seed: int | None = None,
):
    image_reader = ZarrReader(input_path)
    image_metadata = {
        "noise_std": noise_std,
        "add_noise_to_background_only": background_only,
        "noise_seed": seed,
        "raw_image_metadata": dict(image_reader.get_zarr().attrs),
    }
    print(image_metadata)
    image = image_reader.get_dask_array()

    output_writer = ZarrWriter(
        path=output_path,
        shape=image.shape,
        dtype=np.uint8,
        chunks=(128, 128, 128),
        comp_level=0,
        overwrite=True,
        attributes=image_metadata,
    )

    noisy_image = _add_gaussian_noise(
        image=image,
        noise_std=noise_std,
        seed=seed,
        background_only=background_only,
    )

    output_writer.write(noisy_image)



def main():
    pass


if __name__ == "__main__":
    main()
