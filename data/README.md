# Local data

Only this README is tracked. Keep original datasets, extracted particles,
manual labels, and generated volumes here; distribute datasets separately.

The default layout used by `run.py` is:

```text
data/GOH_6_250/
  sim_data_uint8/
    <id>.obj           Particle mesh
    <id>.npy.gz        Matching compressed NumPy volume
  manual_check.json    Labels exported by labeller.py
  outputs/             Generated simulation results and Zarr volumes
```

`manual_check.json` contains a `labels` array. Each selected particle needs a
`file_name` such as `123.npy.gz` and a `label` of `correct`. Use the labelling
UI to produce this file.

The `sim_data_uint8` directory name follows the existing dataset convention;
the extraction command preserves the source volume dtype. You can choose other
locations with `--sim_data_dir`, `--manual_check_json`, and `--output_dir`.
