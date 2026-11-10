"""
This file is used to separate particles in a tomogram.

Workflow:
1. Read the original tomogram and mask,
2. Extract each particle by its mask
3. Save the extracted particles into 3d array and mesh.
"""
import argparse
import csv
import gc
import os
from functools import partial
from multiprocessing import cpu_count, Pool

import numpy as np
from tqdm import tqdm

from preprocess import (get_particle_bounding_boxes,
                        load_particle_bounding_boxes,
                        crop_tomo_mask,
                        voxel_to_mesh,
                        save_particle_voxel_mesh)


class _ParticlePack:
    """
    This class is used to load the metadata of different particle datasets.
    The dataset metadata should be stored in a csv file, where columns are
    - data: the name of the dataset
    - max_labels: maximum number of labels
    - x: size of the dataset in x direction
    - y: size of the dataset in y direction
    - z: size of the dataset in z direction
    """
    def __init__(self, path: str, dataset_name: str = None):
        self._path = path
        self._metadata = None
        self.tomo = None
        self.mask = None
        self.max_labels = 0
        self.shape = None

        self._update_metadata()
        if dataset_name is not None:
            self.name = dataset_name
            self.choose(dataset_name)

    def choose(self, dataset_name):
        """
        Choose a dataset by its name.
        :param dataset_name: name of the dataset
        """
        self.name = dataset_name
        self.shape = (self._metadata[dataset_name]['x'],
                      self._metadata[dataset_name]['y'],
                      self._metadata[dataset_name]['z'])
        self.tomo = os.path.join(self._path, 'tomo', dataset_name, 'tomo.memmap')
        self.mask = os.path.join(self._path, 'mask', dataset_name, 'labels.memmap')
        self.max_labels = self._metadata[dataset_name]['max_labels']

    def _update_metadata(self):
        """
        Reads the metadata, update dataset info.
        """
        metadata_path = os.path.join(self._path, 'metadata.csv')
        with open(metadata_path, 'r') as file:
            reader = csv.DictReader(file)
            self._metadata = {
                row['data']: {
                    key: int(value) for key, value in row.items() if key != 'data'
                } for row in reader
            }

    def get_avail_datasets(self) -> list:
        """
        Get a list of names of all the available datasets
        """
        return list(self._metadata.keys())


def _update_particle_bounding_boxes(out_path: str, particle_pack: _ParticlePack):
    """
    Get the bounding box information of each individual particle in a particle pack
    If there is already been a cached file, use the cached one.
    :param out_path: cached file output path
    :param particle_pack: name of the particle pack
    :return: the bounding boxes of all the particles in a list
    """
    cached_file_path = os.path.join(out_path, f'{particle_pack.name}_bounding_boxes.csv')
    # Load cached file
    if os.path.exists(cached_file_path):
        return load_particle_bounding_boxes(cached_file_path)
    else:
        mask = np.memmap(filename=particle_pack.mask, dtype=np.int16, mode='r', shape=particle_pack.shape)
        bounding_boxes = get_particle_bounding_boxes(mask, particle_pack.max_labels)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.savetxt(cached_file_path, bounding_boxes, delimiter=',', fmt='%i')
        del mask
        gc.collect()
        return bounding_boxes


def _process_save_particle(mask_idx, out_path, particle_pack, bounding_boxes):
    """
    A sub-process that separates and saves a individual particle
    :param mask_idx: index of the mask
    :param out_path: output path
    :param particle_pack: particle pack name
    :param bounding_boxes: a list of bounding boxes
    :return: a return code, -1 means mask not found,
                            0 means program runs normally,
                            1 means there is something wrong with the saving
    """
    if -1 in bounding_boxes[mask_idx]:
        print(f'Mask {mask_idx} not found.')
        return -1
    else:
        try:
            # 1. Load memory map
            tomo = np.memmap(filename=particle_pack.tomo, dtype=np.int16, mode='r', shape=particle_pack.shape)
            mask = np.memmap(filename=particle_pack.mask, dtype=np.int16, mode='r', shape=particle_pack.shape)
            # 2. crop tomogram and mask from memory map by its boudning box
            tomo, mask = crop_tomo_mask(tomo, mask, mask_idx, bounding_boxes)
            # 3. convert mask to mesh
            mesh = voxel_to_mesh(mask)
            # 4. save tomogram as numpy array, save mesh
            save_particle_voxel_mesh(out_path, particle_pack.name, f'{mask_idx + 1:04d}', tomo, mask, mesh)
            # 5. free up memory space
            del tomo, mask, mesh
            gc.collect()
            return 0
        except (Exception, ) as e:
            print(f'An error occurred at mask {mask_idx + 1}: {e}')
            gc.collect()
            return 1


def _process_save_particle_parallel(out_path, particle_pack, bounding_boxes, n_processes):
    """
    A parallel program to separate and save particles
    :param out_path: output path
    :param particle_pack: particle pack name
    :param bounding_boxes: bounding boxes
    :param n_processes: number of processes to process the program in parallel
    """
    sub_task = partial(_process_save_particle, out_path=out_path,
                       particle_pack=particle_pack, bounding_boxes=bounding_boxes)
    if n_processes == 0:
        for i in tqdm(range(particle_pack.max_labels)):
            sub_task(i)
    else:
        with Pool(n_processes, maxtasksperchild=1) as pool:
            run_res = list(tqdm(pool.imap(sub_task, range(particle_pack.max_labels), chunksize=1),
                           total=particle_pack.max_labels, desc=f'Separating Particles'))
        print(f'Processes returned values : {run_res}')


def _parse_args():
    parser = argparse.ArgumentParser(description='A program to separate particles in a tomogram')
    parser.add_argument('--in_dir', type=str, metavar='path', help='input path', required=True)
    parser.add_argument('--pack', type=str, metavar='name', help='particle pack to use', required=True)
    parser.add_argument('--out_dir', type=str, metavar='path', help='output path', required=True)
    parser.add_argument('--voxel_scale', type=float, default=0.5,
                        metavar='float', help=f'rescale the voxel (reduce no. vertices), default={0.5}')
    parser.add_argument('--mesh_scale', type=float, default=0.2,
                        metavar='float', help=f'rescale mesh (NOT reduce no. vertices), default={0.2}')
    parser.add_argument('--n_cores', type=int, default=cpu_count(),
                        metavar='int', help=f'number of cores for parallel processing, default={cpu_count()}')
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    particle_pack = _ParticlePack(args.in_dir, args.pack)
    bounding_boxes = _update_particle_bounding_boxes(args.out_dir, particle_pack)
    _process_save_particle_parallel(args.out_dir, particle_pack, bounding_boxes, args.n_cores)


if __name__ == '__main__':
    main()
