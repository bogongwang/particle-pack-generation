import os
import sys

import numpy as np
from netCDF4 import Dataset
from trimesh.voxel import ops
from trimesh import Trimesh
import scipy
from numba import njit
from numba_progress import ProgressBar
from tqdm import tqdm

import os
import gc
from typing import List, Callable
import json

"""
A Library for preparing particles for simulation
"""


def load_nc(path: str, key: str) -> np.ndarray:
    """
    Load NC file and return its numpy array

    :param path: path to file
    :param key: {tomo, labels}
    :return: stored numpy array in nc file
    """
    return Dataset(path, 'r')[key][:, :, :].data


def tomo_to_int16(arr: np.ndarray) -> np.ndarray:
    """
    Convert a tomogram in numpy array into int16 format by
    - convert less than 0 parts into 0
    - convert dtype into int16

    :param arr: tomogram in numpy array
    """
    return np.array(np.clip(arr, 0, None), dtype=np.int16)


def mask_to_int16(arr: np.ndarray) -> np.ndarray:
    """
    Convert a tomogram in numpy array into int16 format by
    - convert surrounding round mask into 0
    - convert dtype into int16

    :param arr: tomogram in numpy array
    """
    arr[arr == np.iinfo(arr.dtype).max] = 0
    return arr.astype(np.int16)


def merge_nc(in_dir: str, out_dir: str, key: str, end_slice: int, convert_fn: Callable = lambda x: x) -> tuple:
    """
    Merge multiple ncfiles and save it into memory-map
    :param in_dir: input path
    :param out_dir: output path
    :param key: {tomo, labels}
    :param end_slice: end index of slice
    :param convert_fn: conversion function to each individual tomogram
    """
    # get desired shape
    dataset_grid_shape = Dataset(os.path.join(in_dir, f'block00000000.nc'), 'r').total_grid_size_xyz
    # convert to memory-map shape
    mmap_shape = (dataset_grid_shape[2], dataset_grid_shape[0], dataset_grid_shape[1])
    # define memory-map path
    save_path = os.path.join(out_dir, f'{key}.memmap')
    # init memory-map
    mmap = np.memmap(save_path, np.int16, mode='w+', shape=mmap_shape)
    # memory-map start and end index
    mmap_start, mmap_end = 0, 0
    for i in tqdm(range(0, end_slice + 1)):
        filename = os.path.join(in_dir, f'block{i:0>8}.nc')
        curr_nc = convert_fn(load_nc(filename, key))
        mmap_start = mmap_end
        mmap_end = mmap_start + curr_nc.shape[0]
        mmap[mmap_start: mmap_end, :, :] = curr_nc
        mmap.flush()
        del curr_nc
        gc.collect()
    return mmap_shape


def create_dir(parent_path: str, children: List) -> str:
    """
    Create the dataset directory.

    :param parent_path: parent directory
    :param children: children directory to be spawned
    """
    assert type(children) is list

    # create parent directory
    if os.path.isdir(parent_path):
        msg = f'\t[create_dir] EXISTED: "{parent_path}"'
    else:
        os.makedirs(parent_path, exist_ok=False)
        msg = f'\t[create_dir] CREATED: "{parent_path}'

    # create children directories
    for child in children:
        child_path = os.path.join(parent_path, child)
        if os.path.isdir(child_path):
            msg += f'\n\t[create_dir] EXISTED: "{child_path}"'
        else:
            os.makedirs(child_path, exist_ok=False)
            msg += f'\n\t[create_dir] CREATED: "{child_path}"'

    return msg


def get_num_particles(labels: np.ndarray) -> int:
    """
    Get number of particles in labels dataset
    """
    return np.max(labels)


def get_mask_3d(masks: np.ndarray, idx: int):
    """
    Get the 3D-mask of a particle with its index
    :param masks: a mask of all the particles
    :param idx: index of the particle
    :return: the mask of the particle of `idx`
    """
    masks = masks.copy()
    masks[masks != idx] = 0
    masks[masks == idx] = 1
    return masks


def remove_border_3d(arr: np.ndarray) -> np.ndarray:
    """
    Crop out the surrounding empty part of a 3d array
    :param arr: input array
    :return: cropped array
    """
    indices = np.nonzero(arr)

    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]

    x_start, x_end = x_indices.min(), x_indices.max() + 1
    y_start, y_end = y_indices.min(), y_indices.max() + 1
    z_start, z_end = z_indices.min(), z_indices.max() + 1

    return arr[x_start: x_end, y_start: y_end, z_start: z_end]


def crop_3d(arr, bbox) -> np.ndarray:
    """
    :param arr: the 3d array to be cropped
    :param bbox: the bounding box of the object
    :return: the cropped object
    """
    x_start, x_end, y_start, y_end, z_start, z_end = bbox
    return arr[x_start: x_end, y_start: y_end, z_start: z_end]


def get_particle_bounding_boxes(mask, n_labels) -> np.ndarray:
    """
    Get the bounding boxes of all particle masks.
    :param mask: masked particle pack
    :param n_labels: number of labels for particle
    :return: bounding boxes in a 2d array, entry x represents the bounding box of particle named "x+1"
    """
    @njit(nogil=True)
    def _get_particle_bounding_boxes(_mask, _n_labels, _progress_proxy):
        # Initialize an array to hold bounding boxes,
        # setting minimums to inf and maximums to -inf
        # The array shapes n_labels rows, where each row represents
        # [x_min, x_max, y_min, y_max, z_min, z_max]
        bboxes = np.full((_n_labels, 6), np.inf)
        bboxes[:, 1::2] = -np.inf
        # Iterate through the mask to find bounding boxes for each label
        for x in (range(_mask.shape[0])):
            for y in range(_mask.shape[1]):
                for z in range(_mask.shape[2]):
                    label = _mask[x, y, z]
                    if label > 0:  # 0 is background
                        label_index = label - 1  # Adjusting for 0-based indexing
                        x_min, x_max, y_min, y_max, z_min, z_max = bboxes[label_index]
                        # Update the bounding box for the current label
                        bboxes[label_index] = [
                            min(x_min, x), max(x_max, x),
                            min(y_min, y), max(y_max, y),
                            min(z_min, z), max(z_max, z)
                        ]
            _progress_proxy.update(1)
        bboxes[:, 1::2] += 1
        return bboxes
    # Display progress bar
    with ProgressBar(total=mask.shape[0], desc='Getting Bounding Boxes') as progress_proxy:
        bbox_arr = _get_particle_bounding_boxes(mask, n_labels, progress_proxy)
    # Set not founded index to -1
    bbox_arr[np.isinf(bbox_arr)] = -1
    return bbox_arr.astype(np.int16)


def load_particle_bounding_boxes(path) -> np.ndarray:
    """
    Load saved particle bounding boxes.
    :param path: path to the bounding box
    :return: loaded bounding boxes in numpy array
    """
    return np.loadtxt(path, delimiter=',', dtype=np.int16)


def crop_tomo_mask(tomo, mask, mask_index, bound_boxes) -> (np.ndarray, np.ndarray):
    """
    Crop tomographic pack and its mask by giving a bounding box
    :param tomo: tomogram
    :param mask: mask of the tomogram
    :param mask_index: index of the mask
    :param bound_boxes: a list of bounding boxes
    :return: cropped tomogram and its mask
    """
    cropped_mask = np.array(crop_3d(mask, bound_boxes[mask_index]).copy())
    cropped_mask[cropped_mask != mask_index + 1] = 0
    cropped_mask[cropped_mask == mask_index + 1] = 1
    cropped_tomo = np.array(crop_3d(tomo, bound_boxes[mask_index]).copy())
    cropped_tomo *= cropped_mask
    return cropped_tomo, cropped_mask


def voxel_to_mesh(voxel: np.ndarray, voxel_scale: float = 0.5, mesh_scale: float = 0.2) -> Trimesh:
    """
    Convert a voxels in numpy array to mesh
    :param voxel: voxel in numpy array
    :param voxel_scale: scale of the voxel, reduce this will the number of vertices of mesh
    :param mesh_scale: scale of the converted mesh, only affects the scale of the mesh
    :return: converted mesh if successfully convert the voxels to meshes
    """
    voxel = scipy.ndimage.zoom(voxel, voxel_scale)
    mesh = ops.matrix_to_marching_cubes(voxel)
    mesh = mesh.apply_scale([mesh_scale] * 3)
    return mesh


def save_particle_voxel_mesh(out_path: str, pack: str, out_name: str,
                             tomo: np.ndarray, mask: np.ndarray, mesh: Trimesh):
    """
    Save a particle's voxel (tomogram and mask) and mesh to disk
    :param out_path: output path
    :param pack: the pack info
    :param out_name: output of the particle
    :param tomo: tomogram of the particle, save to .npy
    :param mask: mask of the particle, save to .npy
    :param mesh: mesh of the particle, save to .obj
    """
    # Check if the voxel and mesh path exists
    tomo_path = os.path.join(out_path, 'tomo', pack)
    mask_path = os.path.join(out_path, 'mask', pack)
    mesh_path = os.path.join(out_path, 'mesh', pack)
    if not os.path.exists(tomo_path):
        os.makedirs(tomo_path, exist_ok=True)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path, exist_ok=True)
    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path, exist_ok=True)
    np.save(os.path.join(tomo_path, f'{out_name}.npy'), tomo)
    np.save(os.path.join(mask_path, f'{out_name}.npy'), mask)
    mesh.export(os.path.join(mesh_path, f'{out_name}.obj'))

