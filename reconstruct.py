import argparse
import csv
import os
import sys
import json
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm


def load_simulation_result(filename):
    """
    Load simulation result into a dictionary.
    example return:
    >>> {
    >>> 1: [{'name': '01_0001', 'rx': 0.1, 'ry': 0.2, ...}, {'name': '02_0001', 'rx': 0.3, 'ry': 0.4, ...}],
    >>> 2: [{'name': '01_0002', 'rx': 0.4, 'ry': 0.2, ...}, {'name': '02_0001', 'rx': 0.3, 'ry': 0.4, ...}]
    >>> }
    :param filename: name to the simulation file
    :return: loaded dictionary
    """
    result = {}
    with open(filename) as file:
        particle_info_dict = csv.DictReader(file)
        for row in particle_info_dict:
            particle = int(row['particle'][-4:])
            if particle not in result:
                result[particle] = []
            result[particle].append({
                "name": str(row['particle']),
                "rx": float(row["rx"]),
                "ry": float(row["ry"]),
                "rz": float(row["rz"]),
                "tx": float(row["tx"]),
                "ty": float(row["ty"]),
                "tz": float(row["tz"])
            })
    return result


def rotate_voxel(voxel, rotation_angles):
    """
    Rotate a voxel along x, y and z axis
    :param voxel: the voxel to be rotated
    :param rotation_angles: rotation angle in [rx, ry, rz]
    :return: the rotated voxel
    """
    # Apply the rotation transformation around the x-axis
    rotated_volume = rotate(voxel, rotation_angles[0], axes=(1, 2))
    # print(rotated_volume.shape)
    # Apply the rotation transformation around the y-axis
    rotated_volume = rotate(rotated_volume, rotation_angles[1], axes=(0, 2))
    # print(rotated_volume.shape)
    # Apply the rotation transformation around the z-axis
    rotated_volume = rotate(rotated_volume, rotation_angles[2], axes=(0, 1))
    # print(rotated_volume.shape)
    return rotated_volume


def _place_tomo_mask(mmap_tomo, mmap_mask, mmap_shape, mmap_padding, tomo, mask, voxel_valid_areas, location):
    """
    Place tomogram and its mask into the corresponding memory map
    :param mmap_tomo: tomogram memory map
    :param mmap_mask: mask memory map
    :param mmap_shape: original shape of the memory map
    :param mmap_padding: the padding of the memory map
    :param tomo: tomorgram
    :param mask: mask
    :param voxel_valid_areas: valid areas to copy to the corresponding memory map
    :param location: location of the placement
    :return: return value, placement indices, the program will only return placement indices when the placement is
             illegal. There might be two reasons for illegal placement: not enough padding and wrong simulation result
    """
    # Calculate the corner of the subarray where the object will be placed
    # Adjust for scene centre
    padded_mmap_shape = _get_padded_mmap_shape(mmap_shape, mmap_padding)
    scene_center = [mmap_padding[0]] + [s // 2 for s in padded_mmap_shape[1:]]
    # Bottom left corner (the start coordinate of each axis)
    start_coord = [center + loc - voxel_shape // 2 for center, loc, voxel_shape in zip(scene_center,
                                                                                       location,
                                                                                       tomo.shape)]
    placement_indices = [(start, start + voxel_shape) for start, voxel_shape in zip(start_coord, tomo.shape)]
    # Check if placement indices out of bound
    for idx, (start, end) in enumerate(placement_indices):
        if start < 0 or end > padded_mmap_shape[idx]:
            return 1, placement_indices
    # Calculate slices for object placement
    slices = [slice(start, end) for start, end in placement_indices]
    # Place object in scene
    mmap_tomo[slices[0], slices[1], slices[2]][voxel_valid_areas] = tomo[voxel_valid_areas].astype(np.uint16)
    mmap_tomo.flush()
    mmap_mask[slices[0], slices[1], slices[2]][voxel_valid_areas] = mask[voxel_valid_areas].astype(np.uint16)
    mmap_mask.flush()
    return 0, []


def reconstruct(voxel_path: str, pack: str, mmap_path: str, mmap_shape: tuple, mmap_padding: tuple,
                simulation_id: str, simulation_results: dict, simulation_particles: list, index: int):
    """
    - Load tomo and mask voxel
    - Rotate the voxel based on the simulated result
    - Translate the voxel based on the simulated result
    - Place the object based on the simulated result
    :param voxel_path: path to voxels
           (the directory has tomo and mask directory, separated particles are in these directories)
    :param pack: particle pack name
    :param mmap_path: path to memory map
    :param mmap_shape: original shape of the memory map
    :param mmap_padding: padding of the memory map
           the reason to introduce padding is that, after rotation,
           the rotated 3d voxel to be placed may exceed the bound of the target memory map
    :param simulation_id: simulation id (10 digits value)
    :param simulation_results: simulation results (dictionary)
    :param simulation_particles: list of names of particles involved in simulation
    :param index: index of the particle (not the name)
    """
    particle_name = f'{simulation_particles[index - 1]:04d}'
    print(f'{particle_name} started')
    tomo_path = os.path.join(voxel_path, 'tomo', pack, f'{particle_name}.npy')
    mask_path = os.path.join(voxel_path, 'mask', pack, f'{particle_name}.npy')
    tomo = np.load(tomo_path)
    mask = np.load(mask_path)

    padded_mmap_shape = _get_padded_mmap_shape(mmap_shape, mmap_padding)
    mmap_tomo_path = os.path.join(
        mmap_path, 'tomo', pack, simulation_id,
        f'tomo.memmap'
    )
    mmap_mask_path = os.path.join(
        mmap_path, 'mask', pack, simulation_id,
        f'labels.memmap'
    )
    tomo_mmap = np.memmap(mmap_tomo_path, np.int16, mode='r+', shape=padded_mmap_shape)
    mask_mmap = np.memmap(mmap_mask_path, np.int16, mode='r+', shape=padded_mmap_shape)
    for result in simulation_results[simulation_particles[index - 1]]:
        rotation_angles = [result['rx'], -result['ry'], result['rz']]
        translation = np.ceil(10 * np.array([result['tz'], result['tx'], result['ty']])).astype(int)
        # Process mask
        mask[mask > 1] = 1
        rotated_mask = rotate_voxel(mask, rotation_angles).transpose(2, 0, 1)
        valid_area = rotated_mask > 0
        rotated_mask[valid_area] = int(particle_name)
        # Process tomo
        rotated_tomo = rotate_voxel(tomo, rotation_angles).transpose(2, 0, 1)
        # Place tomo and mask
        return_code, return_res = _place_tomo_mask(tomo_mmap, mask_mmap, mmap_shape, mmap_padding,
                                                   rotated_tomo, rotated_mask, valid_area, translation)
        if return_code == 1:
            print(f'Error in processing {result["name"]}, placement={return_res}. '
                  f'Try to increase padding or check the simulation result', file=sys.stderr)
        del rotated_tomo, rotated_mask, valid_area
    del tomo, mask, tomo_mmap, mask_mmap
    print(f'{particle_name} finished')


def _get_padded_mmap_shape(mmap_shape, mmap_padding) -> tuple:
    """
    Calculate the shape of the memory map after padding
    :param mmap_shape: original shape of the memory map
    :param mmap_padding: padding of the memory map
    :return:
    """
    return tuple([original + padding for original, padding in zip(mmap_shape, mmap_padding)])


def _parse_args():
    def _parse_tuple(input_string: str):
        """
        Parse a string representing a tuple of 3 integers.

        Args:
        input_string (str): A string in the format "(int, int, int)".

        Returns:
        tuple: A tuple containing the three integers.
        """

        # Removing left and right parentheses and spaces
        processed_string = (input_string.replace('(', '')
                            .replace(')', '')
                            .replace(' ', ''))

        # Splitting the values by comma and converting them to integers
        split_values = processed_string.split(',')

        # Ensuring that there are exactly 3 elements after splitting
        if len(split_values) != 3:
            raise ValueError("Input string must contain exactly three integers.")

        # Converting the split strings to integers
        try:
            int_values = tuple(map(int, split_values))
        except ValueError:
            raise ValueError("All values in the input string must be integers.")

        return int_values

    parser = argparse.ArgumentParser(description='A program reconstructs tomogram with its mask by simulation result.')
    parser.add_argument('--simulation_path', type=str,
                        metavar='path', help='path to the simulation files')
    parser.add_argument('--simulation_id', type=int,
                        metavar='int', help='id of the simulation, a 10 digit number')
    parser.add_argument('--voxel_path', type=str,
                        metavar='path', help='path to the saved voxels.  '
                                             '(a directory contains `tomo` and `mask` child directory, '
                                             'the saved voxels are in in npy format), ')
    parser.add_argument('--pack', type=str,
                        metavar='str', help='particle pack')
    parser.add_argument('--mmap_path', type=str,
                        metavar='path', help='save path to the memory map')
    parser.add_argument('--mmap_shape', type=str,
                        metavar='(int, int, int)', help='shape of the saved memory map in (z, x, y) order')
    parser.add_argument('--mmap_padding', type=str, default='(100, 100, 100)',
                        metavar='(int, int, int)', help='shape of the padding of the saved memory map '
                                                        'in (z, x, y) order, default=(100, 100, 100)')
    parser.add_argument('--n_cores', type=int, default=cpu_count(),
                        metavar='int', help=f'number of cores for parallel processing, default={cpu_count()}')
    args = parser.parse_args()
    args.mmap_shape = _parse_tuple(str(args.mmap_shape))
    args.mmap_padding = _parse_tuple(str(args.mmap_padding))
    args.simulation_id = str(args.simulation_id)
    return args


def main():
    args = _parse_args()

    simulation_result_path = os.path.join(args.simulation_path, f'simulation_result_{args.simulation_id}.csv')
    simulation_result = load_simulation_result(simulation_result_path)
    simulation_particles = list(simulation_result.keys())

    padded_mmap_shape = _get_padded_mmap_shape(args.mmap_shape, args.mmap_padding)
    mmap_tomo_path = os.path.join(args.mmap_path, 'tomo', args.pack, args.simulation_id, f'tomo.memmap')
    mmap_mask_path = os.path.join(args.mmap_path, 'mask', args.pack, args.simulation_id, f'labels.memmap')
    os.makedirs(os.path.dirname(mmap_tomo_path), exist_ok=True)
    os.makedirs(os.path.dirname(mmap_mask_path), exist_ok=True)
    np.memmap(mmap_tomo_path, np.int16, mode='w+', shape=padded_mmap_shape)
    np.memmap(mmap_mask_path, np.int16, mode='w+', shape=padded_mmap_shape)

    n_subtasks = len(simulation_result)
    subtask = partial(reconstruct, args.voxel_path, args.pack, args.mmap_path, args.mmap_shape, args.mmap_padding,
                      args.simulation_id, simulation_result, simulation_particles)

    with Pool(args.n_cores, maxtasksperchild=1) as pool:
        list(tqdm(pool.imap(subtask, range(1, n_subtasks + 1)),
                  total=n_subtasks,
                  desc='Reconstructing tomogram and mask'))

    with open(os.path.join(args.mmap_path, 'tomo', args.pack, args.simulation_id, 'metadata.json'), 'w') as file:
        data = {
            "name": "tomo.memmap",
            "shape": list(padded_mmap_shape),
            "n_particles": 0,
            "pack": args.pack
        }
        json.dump(data, file, indent=4)
    with open(os.path.join(args.mmap_path, 'mask', args.pack, args.simulation_id, 'metadata.json'), 'w') as file:
        data = {
            "name": "labels.memmap",
            "shape": list(padded_mmap_shape),
            "n_particles": 0,
            "pack": args.pack
        }
        json.dump(data, file, indent=4)
    print(f'\nReconstruction finished.'
          f'\nTomogram saved in \n{mmap_tomo_path}'
          f'\nMask saved in \n{mmap_mask_path}')


if __name__ == '__main__':
    main()
