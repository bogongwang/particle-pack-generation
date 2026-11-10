import argparse
import logging
import os
import time
from itertools import islice
from pathlib import Path

import numpy as np
import dask
import dask.array as da
from numba import njit
from tqdm import tqdm
import trimesh
from trimesh.voxel import ops
from concurrent.futures import ProcessPoolExecutor, as_completed
from dask.diagnostics import ProgressBar

from .io import save_npy_compressed
from pptools.io import ZarrReader, load_nc_arr

logger = logging.getLogger(__name__)


@njit(nogil=True)
def _bbox_one_block(mask_block, n_labels, oz, oy, ox):
    """Calculate label bounding boxes for a single mask chunk.

    Args:
        mask_block: A 3D chunk of the labelled mask.
        n_labels: Highest label ID expected in the mask.
        oz: Global z-axis offset for the chunk.
        oy: Global y-axis offset for the chunk.
        ox: Global x-axis offset for the chunk.

    Returns:
        An array of shape ``(n_labels + 1, 6)`` containing bounding boxes in
        ``[zmin, zmax, ymin, ymax, xmin, xmax]`` order. Missing labels keep the
        sentinel values initialized in this function.
    """
    bboxes = np.empty((n_labels + 1, 6), dtype=np.int32)
    max_int = np.iinfo(np.int32).max

    # Initialize boxes with sentinel values so min/max reductions can merge
    # per-block results without special-casing labels absent from a block.
    for i in range(n_labels + 1):
        bboxes[i, 0] = max_int  # zmin
        bboxes[i, 1] = -1  # zmax
        bboxes[i, 2] = max_int  # ymin
        bboxes[i, 3] = -1  # ymax
        bboxes[i, 4] = max_int  # xmin
        bboxes[i, 5] = -1  # xmax

    for z in range(mask_block.shape[0]):
        for y in range(mask_block.shape[1]):
            for x in range(mask_block.shape[2]):
                label = mask_block[z, y, x]
                if label > 0 and label <= n_labels:
                    gz = z + oz
                    gy = y + oy
                    gx = x + ox

                    if gz < bboxes[label, 0]:
                        bboxes[label, 0] = gz
                    if gz + 1 > bboxes[label, 1]:
                        bboxes[label, 1] = gz + 1

                    if gy < bboxes[label, 2]:
                        bboxes[label, 2] = gy
                    if gy + 1 > bboxes[label, 3]:
                        bboxes[label, 3] = gy + 1

                    if gx < bboxes[label, 4]:
                        bboxes[label, 4] = gx
                    if gx + 1 > bboxes[label, 5]:
                        bboxes[label, 5] = gx + 1

    return bboxes


def get_particle_bounding_boxes(
    mask: da.Array, n_labels: int | None = None
) -> np.ndarray:
    """Calculate bounding boxes for all labels in a chunked 3D mask.

    Args:
        mask: A Dask array containing integer labels in ``(z, y, x)`` order.
        n_labels: Optional highest label ID. If omitted, it is computed from
            the maximum value in ``mask``.

    Returns:
        A NumPy array where ``result[label_id]`` is the bounding box for that
        label in ``[zmin, zmax, ymin, ymax, xmin, xmax]`` order. Labels that do
        not appear in the mask are filled with ``-1``.
    """
    if n_labels is None:
        n_labels = int(mask.max().compute())

    # Convert chunk sizes to global offsets so each delayed block can report
    # bounding boxes in full-volume coordinates.
    z_offsets = np.cumsum((0,) + mask.chunks[0][:-1])
    y_offsets = np.cumsum((0,) + mask.chunks[1][:-1])
    x_offsets = np.cumsum((0,) + mask.chunks[2][:-1])

    partial_bbox = []
    block_grid = mask.to_delayed()

    for iz in range(block_grid.shape[0]):
        for iy in range(block_grid.shape[1]):
            for ix in range(block_grid.shape[2]):
                block = block_grid[iz, iy, ix]

                partial = dask.delayed(_bbox_one_block)(
                    block,
                    n_labels,
                    int(z_offsets[iz]),
                    int(y_offsets[iy]),
                    int(x_offsets[ix]),
                )
                partial_bbox.append(partial)

    with ProgressBar():
        partial_bbox = dask.compute(*partial_bbox)
    partial_bbox = np.stack(partial_bbox, axis=0)

    output_bbox = np.empty((n_labels + 1, 6), dtype=np.int32)

    output_bbox[:, 0] = partial_bbox[:, :, 0].min(axis=0)  # zmin
    output_bbox[:, 1] = partial_bbox[:, :, 1].max(axis=0)  # zmax
    output_bbox[:, 2] = partial_bbox[:, :, 2].min(axis=0)  # ymin
    output_bbox[:, 3] = partial_bbox[:, :, 3].max(axis=0)  # ymax
    output_bbox[:, 4] = partial_bbox[:, :, 4].min(axis=0)  # xmin
    output_bbox[:, 5] = partial_bbox[:, :, 5].max(axis=0)  # xmax

    missing = output_bbox[:, 1] == -1
    output_bbox[missing] = -1

    return output_bbox


def find_labels_from_bbox(bounding_boxes: np.ndarray) -> np.ndarray:
    """Find label IDs that have non-empty bounding boxes.

    Args:
        bounding_boxes: Bounding box array in
        [zmin, zmax, ymin, ymax, xmin, xmax] order, indexed by label ID.

    Returns:
        A 1D array containing label IDs with at least one positive coordinate.
    """
    return np.where((bounding_boxes > 0).any(axis=1))[0]


def extract_objects_3d(
    image: da.Array,
    mask: da.Array,
    labels: np.ndarray,
    bounding_boxes: np.ndarray,
    simp_percent: float,
    simp_max_faces: int,
    output_dir: str,
):
    """Extract labelled objects from a 3D image and save arrays and meshes.

    For each requested label, this function crops the object to its bounding
    box, saves the isolated voxel data as ``.npy``, converts it to a mesh, and
    exports a simplified ``.obj`` file.

    Args:
        image: A 3D image array with shape ``(z, y, x)``.
        mask: A 3D label mask with the same shape as ``image``.
        labels: Label IDs to extract, for example ``[10, 30, 51, 62]``.
        bounding_boxes: Array where ``bounding_boxes[label_id]`` is
            ``[zmin, zmax, ymin, ymax, xmin, xmax]``.
        simp_percent: Fraction of mesh faces to remove during simplification.
            For example, ``0.25`` keeps roughly 75% of the original faces unless
            constrained by ``simp_max_faces``.
        simp_max_faces: Maximum number of faces to keep in the simplified mesh.
        output_dir: Directory where ``.npy`` and ``.obj`` files are written.

    Raises:
        ValueError: If ``image`` and ``mask`` do not have the same shape.
    """

    if image.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape, \
            got image shape={image.shape} and mask shape={mask.shape}")

    labels = labels[labels > 0]
    for label_id in tqdm(labels):
        zmin, zmax, ymin, ymax, xmin, xmax = bounding_boxes[label_id]
        cropped_image = image[zmin:zmax, ymin:ymax, xmin:xmax]
        cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]
        extracted_image = da.where(cropped_mask == label_id, cropped_image, 0).compute(
            # scheduler="threads", num_workers=8
        )
        np.save(os.path.join(output_dir, f"{label_id}.npy"), extracted_image)
        extracted_mesh = ops.matrix_to_marching_cubes(extracted_image)
        # Bound the requested simplification by both the percentage reduction
        # and an absolute face-count cap.
        target_faces = int(
            min(simp_max_faces, extracted_mesh.faces.shape[0] * (1 - simp_percent))
        )
        extracted_mesh_simp = extracted_mesh.simplify_quadric_decimation(
            face_count=target_faces
        )
        extracted_mesh_simp.export(os.path.join(output_dir, f"{label_id}.obj"))


def _extract_obj_arr(
    image: da.Array,
    mask: da.Array,
    label_id: int,
    bounding_box: np.ndarray,
):
    zmin, zmax, ymin, ymax, xmin, xmax = bounding_box
    cropped_image = image[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]
    extracted_image = da.where(cropped_mask == label_id, cropped_image, 0)#.compute(scheduler="threads", num_workers=4)
    return extracted_image

def _anu_int16_to_uint8(arr: np.ndarray, lower: int = 10000, upper=32767) -> np.ndarray:
    """
    Map int16 values in [10000, int16 max] to uint8 values in [0, 255].

    Values below 10000 are clipped to 0.
    Values above int16 max are clipped to 255, though int16 cannot exceed this.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype != np.int16:
        raise ValueError(f"Expected int16 or uint8 array, got {arr.dtype}")
    clipped = np.clip(arr, lower, upper).astype(np.float32)
    scaled = (clipped - lower) * 255.0 / (upper - lower)
    return scaled.astype(np.uint8)

def _simplify_mesh(
    mesh: trimesh.Trimesh, simp_percent: float = 0.9, simp_max_faces: int = 4000
):
    if simp_percent == 0 or simp_max_faces <= 0:
        return mesh
    target_faces = int(min(simp_max_faces, mesh.faces.shape[0] * (1 - simp_percent)))
    target_faces = max(target_faces, 4)
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def _save_extracted_object(
    extracted_image: np.ndarray,
    mesh: trimesh.Trimesh,
    label_id: int,
    output_dir: str,
    comp_level: int = 1,
):
    output_img_path = os.path.join(output_dir, f"{label_id}.npy.gz")
    save_npy_compressed(
        arr=extracted_image, output_path=output_img_path, compress_level=comp_level
    )
    mesh.export(os.path.join(output_dir, f"{label_id}.obj"))
    return label_id


def _process_extracted_obj_arr(
    label_id: int,
    output_dir: str,
    extracted_image: np.ndarray,
    simp_percent: float,
    simp_max_faces: int,
    comp_level: int,
):
    """
    Process one already-extracted labelled object.

    The extraction is intentionally done outside the process pool so that
    Dask image/mask arrays are not repeatedly pickled and sent to workers.
    """
    extracted_mesh = ops.matrix_to_marching_cubes(extracted_image)

    simplified_mesh = _simplify_mesh(
        extracted_mesh,
        simp_percent,
        simp_max_faces,
    )

    saved_image = _anu_int16_to_uint8(extracted_image)
    _save_extracted_object(
        saved_image,
        simplified_mesh,
        label_id,
        output_dir,
        comp_level,
    )

    return label_id


def _iter_batches(iterable, batch_size: int):
    """
    Yield batches from an iterable.
    """
    iterator = iter(iterable)

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def _extract_batch_obj_arrs(
    batch_labels: np.ndarray,
    image: da.Array,
    mask: da.Array,
    bounding_boxes: np.ndarray,
    parallel: int = 16,
):
    """
    Extract one batch of object arrays, computing Dask only once.

    Returns:
        list[(label_id, extracted_image_np)]
    """
    label_ids = []
    extracted_images = []

    for label_id in batch_labels:
        bounding_box = bounding_boxes[label_id]

        extracted_image = _extract_obj_arr(
            image,
            mask,
            label_id,
            bounding_box,
        )

        label_ids.append(label_id)
        extracted_images.append(extracted_image)

    computed_images = dask.compute(
        *extracted_images,
        scheduler="threads",
        num_workers=parallel,
    )

    return list(zip(label_ids, computed_images))


def to_sim_data(
    output_dir: str,
    image: da.Array,
    mask: da.Array,
    labels: np.ndarray,
    bounding_boxes: np.ndarray,
    simp_percent: float = 0.9,
    simp_max_faces: int = 4000,
    comp_level: int = 1,
    parallel: int = 8,
    batch_size: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    labels = labels[labels > 0]

    if batch_size is None:
        batch_size = max(1, parallel)

    if parallel <= 1:
        for batch_labels in tqdm(
            _iter_batches(labels, batch_size),
            total=(len(labels) + batch_size - 1) // batch_size,
            desc="Creating simulation data",
        ):
            extracted_batch = _extract_batch_obj_arrs(
                batch_labels,
                image,
                mask,
                bounding_boxes,
            )

            for label_id, extracted_image in extracted_batch:
                _process_extracted_obj_arr(
                    label_id,
                    output_dir,
                    extracted_image,
                    simp_percent,
                    simp_max_faces,
                    comp_level,
                )

        return

    total_batches = (len(labels) + batch_size - 1) // batch_size

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        for batch_labels in tqdm(
            _iter_batches(labels, batch_size),
            total=total_batches,
            desc="Extracting batches",
        ):
            # Serial extraction in the main process.
            extracted_batch = _extract_batch_obj_arrs(
                batch_labels,
                image,
                mask,
                bounding_boxes,
                parallel=parallel,
            )

            # Parallel processing for this batch only.
            futures = [
                executor.submit(
                    _process_extracted_obj_arr,
                    label_id,
                    output_dir,
                    extracted_image,
                    simp_percent,
                    simp_max_faces,
                    comp_level,
                )
                for label_id, extracted_image in extracted_batch
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing batch in parallel",
                leave=False,
            ):
                future.result()


def create_simulation_data(
    output_dir: str,
    image: da.Array,
    mask: da.Array,
    n_labels: int = 5000,
    simp_percent: float = 0.9,
    simp_max_faces: int = 4000,
    comp_level: int = 1,
    parallel: int = 8,
    batch_size: int | None = None,
):
    boxes = get_particle_bounding_boxes(mask, n_labels=n_labels)
    labels = find_labels_from_bbox(boxes)
    to_sim_data(
        output_dir=output_dir,
        image=image,
        mask=mask,
        labels=labels,
        bounding_boxes=boxes,
        simp_percent=simp_percent,
        simp_max_faces=simp_max_faces,
        comp_level=comp_level,
        parallel=parallel,
        batch_size=batch_size,
    )


def _load_array(input_path: str) -> da.Array:
    suffix = Path(input_path.rstrip(os.sep)).suffix.lower()

    if suffix == ".nc":
        logger.info(f"Loading NetCDF array from {input_path}")
        return load_nc_arr(input_path)

    if suffix == ".zarr":
        logger.info(f"Loading Zarr array from {input_path}")
        return ZarrReader(input_path).get_dask_array()

    raise ValueError(
        f"Unsupported input extension '{suffix}' for {input_path}. "
        "Expected a .nc file/pattern or .zarr store."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create particle simulation data from tomogram and mask arrays."
    )
    parser.add_argument(
        "--tomo_path",
        type=str,
        required=True,
        help="Path to the tomogram input. Supports .nc files/patterns and .zarr stores.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the mask input. Supports .nc files/patterns and .zarr stores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where simulation data files will be written.",
    )
    parser.add_argument(
        "--n_labels",
        type=int,
        default=5000,
        help="Number of labels to process. Labels are assumed to be in the range [1, n_labels]."
    )
    parser.add_argument(
        "--simp_percent",
        type=float,
        default=0.9,
        help="Fraction of mesh faces to remove during simplification.",
    )
    parser.add_argument(
        "--simp_max_faces",
        type=int,
        default=4000,
        help="Maximum number of faces to keep in simplified meshes.",
    )
    parser.add_argument(
        "--comp_level",
        type=int,
        default=1,
        help="Compression level for saved object arrays.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of worker processes used for mesh processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of labels to extract per batch. Defaults to 64",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading input arrays.")
    tomo = _load_array(args.tomo_path)
    mask = _load_array(args.mask_path)

    logger.info(f"Loaded tomo shape={tomo.shape} dtype={tomo.dtype}")
    logger.info(f"Loaded mask shape={mask.shape} dtype={mask.dtype}")

    logger.info(f"Creating simulation data in {args.output_dir}")
    create_simulation_data(
        output_dir=args.output_dir,
        image=tomo,
        mask=mask,
        n_labels=args.n_labels,
        simp_percent=args.simp_percent,
        simp_max_faces=args.simp_max_faces,
        comp_level=args.comp_level,
        parallel=args.parallel,
        batch_size=args.batch_size,
    )
    logger.info(f"Finished creating simulation data in {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    main()
