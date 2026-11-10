# Synthetic Particle Pack Generation

Official code base for paper [Synthetic particle pack generation for augmentation and testing in geological tomographic segmentation](https://doi.org/10.1016/j.tmater.2025.100072).

### TL;DR
**Synthetic particle pack generation** is designed for augmenting datasets and benchmarking geological tomographic segmentation models.
The method generates synthetic 3D tomograms along with corresponding segmentation ground truths.

## 🔥 Latest updates

**November 2026:** The Newton Physics workflow is now included in this repository,
with GPU simulation, Zarr volume reconstruction, a particle review UI, and a
batch generation entry point.
Dependencies are now managed with `uv`, and research
notebooks are organized under `notebooks/`.

The overall workflow remains unchanged from the original method.
The implementation has been updated to use GPU acceleration with Newton Physics, delivering up to a **10×** speedup over the original Blender implementation.

**May 2026:** Rewrote the particle simulation workflow using
[Newton Physics](https://github.com/newton-physics/newton) for GPU acceleration.

## 📦 Example datasets

Example generated datasets, containing synthetic tomograms and corresponding
segmentation masks, are available on [Hugging Face](https://huggingface.co/datasets/bogongwang/synthetic-particle-pack).


## 🎬 Demo

### Simulation process

https://github.com/user-attachments/assets/4bc16975-8540-4d1c-9665-53e6ea6df213

### Generated tomogram-mask pairs

|          | Example 1 | Example 2 | Example 3 | Example 4 | Example 5 | Example 6 |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Tomogram | ![tomo_1714213866_600](https://github.com/user-attachments/assets/946d9739-53c3-4fed-85c1-3e8ed5f06ae8) | ![tomo_1714213866_900](https://github.com/user-attachments/assets/e054c578-4acc-4eaa-8e8f-3b12f1ff4e81) | ![tomo_1714213968_600](https://github.com/user-attachments/assets/b1835783-8e7c-4ac7-b2e6-97ce18028421) | ![tomo_1714213968_900](https://github.com/user-attachments/assets/0de3147c-bbe9-4dcd-8dca-dec5aa22dd30) | ![tomo_1714213978_600](https://github.com/user-attachments/assets/46c09e16-6e93-4eb3-9b7c-28b886fbc44f) | ![tomo_1714213978_900](https://github.com/user-attachments/assets/a44ca867-9fe6-4bc0-b72d-22c514c0a8c4) |
|   Mask   | ![mask_1714213866_600](https://github.com/user-attachments/assets/9df75809-724c-45cb-b256-7b4f7521f594) | ![mask_1714213866_900](https://github.com/user-attachments/assets/4449265b-11d7-4a51-a7e6-915e66fae2c7) | ![mask_1714213968_600](https://github.com/user-attachments/assets/63d1019b-1cd4-41f4-bbaa-98526c802d2a) | ![mask_1714213968_900](https://github.com/user-attachments/assets/4261fc89-b250-4df0-8465-67c92da2bca9) | ![mask_1714213978_600](https://github.com/user-attachments/assets/7c9b86d0-d8fd-47bf-b3e1-c4d5831a8a7d) | ![mask_1714213978_900](https://github.com/user-attachments/assets/882cc72a-be77-4fe9-a3ba-05714f64d7b4) |

## 🛠️ Setup

Use Python 3.12 or newer and `uv`. Run commands from the repository root.
The current particle SDF construction requires a CUDA-capable NVIDIA GPU and
compatible driver, including when simulation is configured with `--device cpu`.
The default simulation device is `cuda:0`. Data preparation and result inspection
can run without a GPU.

```bash
uv sync --locked
uv run run.py --help
```

Install optional tools when needed:

```bash
uv sync --locked --extra labeller --extra notebooks
```

## ⚙️ Newton workflow

Input datasets and generated files stay under `data/` and are excluded from Git.
See [data/README.md](data/README.md) for the expected directory layout.
The original tomograms and segmentation masks must be supplied separately.

1. Extract numbered particle meshes (`<id>.obj`) and compressed volumes
   (`<id>.npy.gz`) from matching tomogram and mask arrays. Inputs can be Zarr
   stores or quoted NetCDF file patterns:

   ```bash
   uv run python -m src.create_simulation_data_zarr \
     --tomo_path data/GOH_6_250/tomo.zarr \
     --mask_path data/GOH_6_250/mask.zarr \
     --output_dir data/GOH_6_250/sim_data_uint8
   ```

   The output directory name does not convert the input dtype. Prepare the
   tomogram's intensity range/dtype as needed before extraction; see
   `notebooks/prepare_data.ipynb` for data preparation.

2. Review the particles with the labelling UI:

   ```bash
   uv run --extra labeller python labeller.py
   ```

   Open `http://localhost:7888`. Set the input directory to
   `data/GOH_6_250/sim_data_uint8` and the output JSON path to
   `data/GOH_6_250/manual_check.json`, then label usable particles as `correct`.
   The UI listens on `0.0.0.0:7888`.

3. Generate synthetic packs, tomograms, and masks:

   ```bash
   uv run run.py \
     --sim_data_dir data/GOH_6_250/sim_data_uint8 \
     --manual_check_json data/GOH_6_250/manual_check.json \
     --output_dir data/GOH_6_250/outputs/example \
     --n_objects 500 \
     --n_simulations 5 \
     --seed 42
   ```

   Each simulation writes its particle transforms to `sim_res.json`. Successful
   simulations also write `tomo.zarr`, `mask.zarr`, and `tomo_noisy.zarr`.
   Without `--output_dir`, results go to a timestamped directory under
   `data/GOH_6_250/outputs/`. See `uv run run.py --help` for size proportions,
   device selection, and worker settings.

## 📁 Repository layout

```text
run.py                  Batch generation entry point
labeller.py             Gradio particle review UI
src/                    Extraction, simulation, and volume reconstruction
notebooks/              Three curated workflow examples
data/                   Local datasets and outputs (only README is tracked)
pyproject.toml          Project dependencies and optional tools
uv.lock                 Locked dependency versions
legacy/                 Original Blender workflow scripts
```

## 📓 Notebooks

Launch with `uv run --extra notebooks jupyter lab` from the repository root.
Run the examples in this order, editing their configuration cells for your data:

| Notebook | Purpose |
| --- | --- |
| [prepare_data.ipynb](notebooks/prepare_data.ipynb) | Crop matching inputs, convert intensities, save Zarr arrays, and extract particles. |
| [simulation_demo.ipynb](notebooks/simulation_demo.ipynb) | Simulate reviewed particles, optionally view their motion, and reconstruct volumes. |
| [inspect_results.ipynb](notebooks/inspect_results.ipynb) | Compare tomograms and masks, with optional PNG and TIFF slice export. |

Source datasets are supplied separately. After preparation, review particles
with `uv run --extra labeller python labeller.py` and save `manual_check.json`
before running the simulation example. Each notebook locates the repository
root and uses configurable paths. Cells that write data describe their outputs
and overwrite behavior.

Original research notebooks remain available locally but are excluded from Git;
they are not part of the supported examples. Clear outputs before committing
notebooks. Datasets, generated results, environments, and local settings are
also ignored.

## 🕰️ Original Blender workflow

The original scripts are preserved in `legacy/` for reproducing the
earlier workflow. They use Blender and a separate environment; the `uv`
setup above describes the Newton workflow.


1. Convert tomogram & segmentation mask to `numpy` memory map (in case out-of-memory issue)
2. Extract particles & convert them into meshes
   - Run `legacy/separate.py` to extract each particle and its mask so that we can convert them into meshes.
3. Simulate particle physics in Blender
   - Use `legacy/simulate.py` to perform physical simulations in [Blender](https://www.blender.org).
4. Generate synthetic tomogram
   - Use `legacy/reconstruct.py` to reconstruct tomograms with the simulated placements and rotations.
5. Post-process


Run the original commands from the repository root using the legacy paths:

```bash
python3 legacy/separate.py --help
python3 legacy/reconstruct.py --help
blender --background --python legacy/simulate.py -- --help
```

Keep input and output paths relative to the repository root, or use absolute
paths. `legacy/preprocess.py` and `legacy/utils.py` provide the original
preprocessing and visualization helpers.

The example below illustrate the original Blender workflow.
https://github.com/user-attachments/assets/a5b87a04-eabd-4149-a761-f11d7f94f292
