# Synthetic Particle Pack Generation

Official code base for paper "[Synthetic particle pack generation for augmentation and testing in geological tomographic segmentation](https://doi.org/10.1016/j.tmater.2025.100072)".

**Synthetic particle pack generation** is designed for augmenting datasets and benchmarking geological tomographic segmentation models. The method generates synthetic 3D tomograms along with corresponding segmentation ground truths.

## Latest updates

**May 2026:** We have rewritten the entire workflow with [Newton Physics](https://github.com/newton-physics/newton) for GPU acceleration.
We will release the updated workflow soon.

## Demo

### Simulation process


https://github.com/user-attachments/assets/a5b87a04-eabd-4149-a761-f11d7f94f292


### Generated tomogram-mask pairs

|          | Example 1 | Example 2 | Example 3 | Example 4 | Example 5 | Example 6 |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Tomogram | ![tomo_1714213866_600](https://github.com/user-attachments/assets/946d9739-53c3-4fed-85c1-3e8ed5f06ae8) | ![tomo_1714213866_900](https://github.com/user-attachments/assets/e054c578-4acc-4eaa-8e8f-3b12f1ff4e81) | ![tomo_1714213968_600](https://github.com/user-attachments/assets/b1835783-8e7c-4ac7-b2e6-97ce18028421) | ![tomo_1714213968_900](https://github.com/user-attachments/assets/0de3147c-bbe9-4dcd-8dca-dec5aa22dd30) | ![tomo_1714213978_600](https://github.com/user-attachments/assets/46c09e16-6e93-4eb3-9b7c-28b886fbc44f) | ![tomo_1714213978_900](https://github.com/user-attachments/assets/a44ca867-9fe6-4bc0-b72d-22c514c0a8c4) |
|   Mask   | ![mask_1714213866_600](https://github.com/user-attachments/assets/9df75809-724c-45cb-b256-7b4f7521f594) | ![mask_1714213866_900](https://github.com/user-attachments/assets/4449265b-11d7-4a51-a7e6-915e66fae2c7) | ![mask_1714213968_600](https://github.com/user-attachments/assets/63d1019b-1cd4-41f4-bbaa-98526c802d2a) | ![mask_1714213968_900](https://github.com/user-attachments/assets/4261fc89-b250-4df0-8465-67c92da2bca9) | ![mask_1714213978_600](https://github.com/user-attachments/assets/7c9b86d0-d8fd-47bf-b3e1-c4d5831a8a7d) | ![mask_1714213978_900](https://github.com/user-attachments/assets/882cc72a-be77-4fe9-a3ba-05714f64d7b4) |

## Implementation & Usage

1. Convert tomogram & segmentation mask to `numpy` memory map (in case out-of-memory issue)
2. Extract particles & convert them into meshes
   - Run `separate.py` to extract each particle and its mask so that we can convert them into meshes.
3. Simulate particle physics in Blender
   - Use `simulate.py` to perform physical simulations in [Blender](https://www.blender.org).
4. Generate synthetic tomogram
   - Use `reconstruct.py` to reconstruct tomograms with the simulated placements and rotations.
5. Post-process

## Additional Notes

- Follow the workflow above to run the code locally.  
- Use `python3 program.py --help` to view available commands.  
  - Example: `python3 reconstruct.py --help`.  
