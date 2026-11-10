
import gzip
import numpy as np


def save_npy_compressed(arr, output_path, compress_level=1):
    with gzip.open(output_path, 'wb', compresslevel=compress_level) as f:
        np.save(f, arr)


def load_npy_compressed(input_path):
    with gzip.open(input_path, 'rb') as f:
        arr = np.load(f)
    return arr
