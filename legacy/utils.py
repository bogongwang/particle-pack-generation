import json
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors


class ParticlePackDataset:
    def __init__(self, tomo_path: str, mask_path: str,
                 z_slice: slice = slice(None), x_slice: slice = slice(None), y_slice: slice = slice(None)):
        """
        :param tomo_path: path to the tomogram
        :param mask_path: path to the mask
        :param z_slice: tomogram and mask range in Z axis
        :param x_slice: tomogram and mask range in X axis
        :param y_slice: tomogram and mask range in Y axis
        """
        self.tomo_path = tomo_path
        self.mask_path = mask_path
        self.z_slice = z_slice
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.tomo = ParticlePackDataset.load_memmap(tomo_path)[z_slice, x_slice, y_slice]
        self.mask = ParticlePackDataset.load_memmap(mask_path)[z_slice, x_slice, y_slice]

    @staticmethod
    def load_memmap(path):
        with open(os.path.join(path, 'metadata.json'), 'r') as file:
            metadata = json.load(file)
            mmap = np.memmap(os.path.join(path, metadata['name']), dtype=np.int16, mode='r',
                             shape=tuple(metadata['shape']))
            return mmap

    @staticmethod
    def img_int16_to_uint8(img):
        shifted_img = img - np.min(img)
        scale_factor = 255.0 / np.max(shifted_img)
        img_uint8 = (shifted_img * scale_factor).astype(np.uint8)
        return img_uint8


class Visualiser:
    def __init__(self, cmap='gist_rainbow', n_colors=32768, seed=42, latex=False):
        if latex:
            text_color = 'black'
            plt.style.use('default')
            plt.rcParams['text.color'] = text_color
            plt.rcParams['axes.labelcolor'] = text_color
            plt.rcParams['xtick.color'] = text_color
            plt.rcParams['ytick.color'] = text_color
            plt.rcParams['axes.edgecolor'] = text_color
            plt.rcParams['font.family'] = 'Computer Modern'
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['text.usetex'] = True
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        else:
            mpl.rcParams.update(mpl.rcParamsDefault)
        self.tomo_cmap = 'gray'
        self.mask_cmap = Visualiser.create_mask_cmap(n_colors, cmap=cmap, seed=seed)
        self.mask_cmap_norm = mcolors.BoundaryNorm(np.arange(0, n_colors), self.mask_cmap.N)

    def plot_tomo(self, tomo, ax=None):
        if ax is None:
            plt.gca().set_axis_off()
            plt.imshow(tomo, cmap=self.tomo_cmap, interpolation='none')
        else:
            plt.gca().set_axis_off()
            ax.imshow(tomo, cmap=self.tomo_cmap, interpolation='none', rasterized=True)

    def plot_mask(self, mask, ax=None):
        if ax is None:
            plt.gca().set_axis_off()
            plt.imshow(mask, cmap=self.mask_cmap, norm=self.mask_cmap_norm, interpolation='none', rasterized=True)
        else:
            plt.gca().set_axis_off()
            ax.imshow(mask, cmap=self.mask_cmap, norm=self.mask_cmap_norm, interpolation='none', rasterized=True)

    def compare_tomo_mask(self, tomo, mask, figsize=(10, 5)):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        self.plot_tomo(tomo, ax[0])
        self.plot_mask(mask, ax[1])
        plt.show()

    def compare_masks(self, mask1, mask2, mask3=None, mask1_title=None, mask2_title=None, mask3_title=None, figsize=(10, 5)):
        ncols = 2 if mask3 is None else 3
        fig, ax = plt.subplots(1, ncols, figsize=figsize)
        ax[0].axis("off")
        ax[1].axis("off")

        self.plot_mask(mask1, ax[0])
        ax[0].set_title(mask1_title)
        self.plot_mask(mask2, ax[1])
        ax[1].set_title(mask2_title)
        if mask3 is not None:
            self.plot_mask(mask3, ax[2])
            ax[2].set_title(mask3_title)
        # plt.show()

    def compare_tomo_mask_pred(self, tomo, mask, pred):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        self.plot_tomo(tomo, ax[0])
        self.plot_mask(mask, ax[1])
        self.plot_mask(pred, ax[2])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        plt.show()

    def show(self):
        plt.show()

    @staticmethod
    def create_mask_cmap(n, cmap='gist_rainbow', seed: int = None):
        """
        Create a custom color map that plots n colors, while leaving the background black
        """
        original_cmap = colormaps[cmap]
        colors = np.linspace(0, 1, n - 1)
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(colors)
        cmap_colors = original_cmap(colors)
        black = np.array([[0, 0, 0, 1]])
        cmap_colors = np.concatenate((black, cmap_colors))
        return ListedColormap(cmap_colors)
