from __future__ import annotations

import datetime
import gzip
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np


AXES = ("z", "y", "x")
AXIS_TO_DIM = {"z": 0, "y": 1, "x": 2}
VIEWER_HEIGHT = 520


CSS = f"""
#media-row {{
    align-items: stretch;
}}

#slice-viewer,
#mesh-viewer {{
    height: {VIEWER_HEIGHT}px;
}}

#slice-viewer img {{
    object-fit: contain;
}}

#mesh-status p {{
    margin: 0;
}}

#correct-button button {{
    background: #15803d;
    border-color: #15803d;
    color: white;
}}

#correct-button button:hover {{
    background: #166534;
    border-color: #166534;
}}
"""


def load_npy_compressed(input_path):
    with gzip.open(input_path, "rb") as f:
        arr = np.load(f)
    return arr


def natural_key(path: str) -> tuple[int, int | str]:
    name = Path(path).name.removesuffix(".npy.gz")
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def discover_volume_files(input_dir: str) -> list[str]:
    directory = Path(input_dir).expanduser()
    if not directory.is_dir():
        raise gr.Error(f"Input directory does not exist: {directory}")
    return sorted(
        [str(path) for path in directory.glob("*.npy.gz")],
        key=natural_key,
    )


def current_timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_labels(output_file: str) -> tuple[dict[str, str], dict[str, str]]:
    path = Path(output_file).expanduser()
    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        labels = {}
        timestamps = {}
        for item in data["labels"]:
            if not isinstance(item, dict) or "file_name" not in item or "label" not in item:
                continue
            file_name = str(item["file_name"])
            labels[file_name] = str(item["label"])
            timestamp = item.get("timestamp") or item.get("time")
            if timestamp:
                timestamps[file_name] = str(timestamp)
        return labels, timestamps
    if isinstance(data, dict):
        return {str(file_name): str(label) for file_name, label in data.items()}, {}
    if isinstance(data, list):
        labels = {}
        timestamps = {}
        for item in data:
            if not isinstance(item, dict) or "file_name" not in item or "label" not in item:
                continue
            file_name = str(item["file_name"])
            labels[file_name] = str(item["label"])
            timestamp = item.get("timestamp") or item.get("time")
            if timestamp:
                timestamps[file_name] = str(timestamp)
        return labels, timestamps

    raise gr.Error(f"Could not read labels from unsupported JSON format: {path}")


def save_labels(output_file: str, labels: dict[str, str], timestamps: dict[str, str]) -> None:
    path = Path(output_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": [
            {
                "file_name": file_name,
                "label": label,
                "timestamp": timestamps.get(file_name),
            }
            for file_name, label in sorted(labels.items(), key=lambda item: natural_key(item[0]))
        ]
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(temp_path, path)


def mesh_path_for_volume(volume_path: str) -> tuple[str | None, str]:
    path = Path(volume_path)
    mesh_path = path.with_name(path.name.removesuffix(".npy.gz") + ".obj")
    if not mesh_path.exists():
        return None, f"No mesh found for {path.name}."

    cache_dir = mesh_path.parent / ".labeller_mesh_cache"
    glb_path = cache_dir / f"{mesh_path.stem}.glb"
    try:
        if not glb_path.exists() or glb_path.stat().st_mtime < mesh_path.stat().st_mtime:
            import trimesh

            cache_dir.mkdir(exist_ok=True)
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.export(glb_path)
        return str(glb_path), f"Mesh: {mesh_path.name}"
    except Exception as exc:
        return str(mesh_path), f"Mesh: {mesh_path.name} (OBJ fallback; GLB conversion failed: {exc})"


def middle_slice_index(volume: np.ndarray, axis: str) -> int:
    return volume.shape[AXIS_TO_DIM[axis]] // 2


def to_display_image(slice_2d: np.ndarray) -> np.ndarray:
    image = np.asarray(slice_2d)
    if image.ndim != 2:
        image = np.squeeze(image)
    if image.dtype == np.uint8:
        return image

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(np.min(image[finite]))
    max_value = float(np.max(image[finite]))
    if max_value == min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def slice_volume(volume: np.ndarray | None, axis: str, slice_index: int) -> np.ndarray | None:
    if volume is None:
        return None
    dim = AXIS_TO_DIM[axis]
    slice_index = int(np.clip(slice_index, 0, volume.shape[dim] - 1))
    if axis == "z":
        return to_display_image(volume[slice_index, :, :])
    if axis == "y":
        return to_display_image(volume[:, slice_index, :])
    return to_display_image(volume[:, :, slice_index])


def object_info_text(
    files: list[str],
    labels: dict[str, str],
    index: int,
    shape: tuple[int, ...] | None = None,
) -> str:
    total = len(files)
    labelled = sum(1 for path in files if Path(path).name in labels)
    if total == 0:
        return "No objects loaded."

    correct_count = sum(1 for label in labels.values() if label == "correct")
    incorrect_count = sum(1 for label in labels.values() if label == "incorrect")
    file_name = Path(files[index]).name
    mesh_name = file_name.removesuffix(".npy.gz") + ".obj"
    status = "Labelled" if file_name in labels else "Unlabelled"
    shape_text = f"Shape: {shape}" if shape is not None else "Shape: unknown"
    return (
        f"Labelled {labelled} / {total} objects. Correct: {correct_count}. Incorrect: {incorrect_count}.\n\n"
        f"**{status}** | `{file_name}` | `{mesh_name}` | {shape_text}"
    )


def first_unlabelled_index(files: list[str], labels: dict[str, str]) -> int:
    for index, path in enumerate(files):
        if Path(path).name not in labels:
            return index
    return 0


def next_unlabelled_index(files: list[str], labels: dict[str, str], current_index: int) -> int:
    if not files:
        return 0
    for offset in range(1, len(files) + 1):
        index = (current_index + offset) % len(files)
        if Path(files[index]).name not in labels:
            return index
    return current_index


def adjacent_index(files: list[str], current_index: int, direction: int) -> int:
    if not files:
        return 0
    return (current_index + direction) % len(files)


def load_current_object(
    files: list[str],
    labels: dict[str, str],
    index: int,
    axis: str,
    requested_slice: int | None = None,
) -> tuple[Any, ...]:
    if not files:
        return (
            None,
            None,
            gr.update(minimum=0, maximum=1, value=0, interactive=False),
            "No .npy.gz files found.",
            None,
            "",
            object_info_text(files, labels, index),
        )

    volume_path = files[index]
    volume = load_npy_compressed(volume_path)
    if volume.ndim != 3:
        raise gr.Error(f"Expected a 3D array, got shape {volume.shape}: {volume_path}")

    max_slice = volume.shape[AXIS_TO_DIM[axis]] - 1
    slider_max = max(1, max_slice)
    if requested_slice is None:
        requested_slice = middle_slice_index(volume, axis)
    slice_index = int(np.clip(requested_slice, 0, max_slice))
    file_name = Path(volume_path).name
    current_label = labels.get(file_name, "unlabelled")
    title = f"{file_name} ({current_label})"
    mesh_path, mesh_status = mesh_path_for_volume(volume_path)

    return (
        volume,
        slice_volume(volume, axis, slice_index),
        gr.update(minimum=0, maximum=slider_max, value=slice_index, interactive=True),
        title,
        mesh_path,
        mesh_status,
        object_info_text(files, labels, index, volume.shape),
    )


def load_dataset(input_dir: str, output_file: str, axis: str):
    if not output_file:
        raise gr.Error("Choose an output JSON file path.")

    files = discover_volume_files(input_dir)
    labels, timestamps = load_labels(output_file)
    index = first_unlabelled_index(files, labels)
    volume, image, slider, title, mesh_path, mesh_status, progress = load_current_object(files, labels, index, axis)
    return files, labels, timestamps, index, volume, image, slider, title, mesh_path, mesh_status, progress


def update_axis(axis: str, slice_index: int, volume: np.ndarray | None):
    if volume is None:
        return None, gr.update(minimum=0, maximum=1, value=0, interactive=False)

    max_slice = volume.shape[AXIS_TO_DIM[axis]] - 1
    slider_max = max(1, max_slice)
    slice_index = middle_slice_index(volume, axis)
    return slice_volume(volume, axis, slice_index), gr.update(maximum=slider_max, value=slice_index)


def update_slice(axis: str, slice_index: int, volume: np.ndarray | None):
    return slice_volume(volume, axis, slice_index)


def mark_label(
    label: str,
    files: list[str],
    labels: dict[str, str],
    timestamps: dict[str, str],
    index: int,
    output_file: str,
    axis: str,
):
    if not files:
        raise gr.Error("Load a directory before labelling.")

    file_name = Path(files[index]).name
    labels = dict(labels)
    timestamps = dict(timestamps)
    labels[file_name] = label
    timestamps[file_name] = current_timestamp()
    save_labels(output_file, labels, timestamps)

    next_index = next_unlabelled_index(files, labels, index)
    volume, image, slider, title, mesh_path, mesh_status, progress = load_current_object(
        files,
        labels,
        next_index,
        axis,
    )
    return labels, timestamps, next_index, volume, image, slider, title, mesh_path, mesh_status, progress


def navigate_object(
    direction: int,
    files: list[str],
    labels: dict[str, str],
    index: int,
    axis: str,
):
    if not files:
        raise gr.Error("Load a directory before navigating.")

    next_index = adjacent_index(files, index, direction)
    volume, image, slider, title, mesh_path, mesh_status, progress = load_current_object(
        files,
        labels,
        next_index,
        axis,
    )
    return next_index, volume, image, slider, title, mesh_path, mesh_status, progress


with gr.Blocks(title="3D Image Labeller") as demo:
    gr.Markdown("# 3D Image Labeller")

    files_state = gr.State([])
    labels_state = gr.State({})
    timestamps_state = gr.State({})
    index_state = gr.State(0)
    volume_state = gr.State(None)

    with gr.Row():
        input_dir = gr.Textbox(label="Input directory", placeholder="/path/to/folder")
        output_file = gr.Textbox(label="Output JSON file", value="labels.json")
        load_button = gr.Button("Load", variant="primary")

    progress = gr.Markdown("No objects loaded.")
    title = gr.Markdown("", visible=False)
    mesh_status = gr.Markdown("", visible=False, elem_id="mesh-status")

    with gr.Row(elem_id="media-row"):
        image = gr.Image(
            label="Slice view",
            type="numpy",
            image_mode="L",
            height=VIEWER_HEIGHT,
            elem_id="slice-viewer",
        )
        mesh = gr.Model3D(label="3D mesh", height=VIEWER_HEIGHT, elem_id="mesh-viewer")

    with gr.Row():
        axis = gr.Radio(choices=list(AXES), value="z", label="Axis", scale=1)
        slice_slider = gr.Slider(
            minimum=0,
            maximum=1,
            step=1,
            value=0,
            label="Slice",
            interactive=False,
            scale=3,
        )

    with gr.Row():
        previous_button = gr.Button("Previous", scale=1)
        next_button = gr.Button("Next", scale=1)
        correct_button = gr.Button("Correct", variant="primary", scale=1, elem_id="correct-button")
        incorrect_button = gr.Button("Incorrect", variant="stop", scale=1)

    load_button.click(
        load_dataset,
        inputs=[input_dir, output_file, axis],
        outputs=[
            files_state,
            labels_state,
            timestamps_state,
            index_state,
            volume_state,
            image,
            slice_slider,
            title,
            mesh,
            mesh_status,
            progress,
        ],
    )
    axis.change(update_axis, inputs=[axis, slice_slider, volume_state], outputs=[image, slice_slider])
    slice_slider.change(update_slice, inputs=[axis, slice_slider, volume_state], outputs=image)
    previous_button.click(
        lambda *args: navigate_object(-1, *args),
        inputs=[files_state, labels_state, index_state, axis],
        outputs=[index_state, volume_state, image, slice_slider, title, mesh, mesh_status, progress],
    )
    next_button.click(
        lambda *args: navigate_object(1, *args),
        inputs=[files_state, labels_state, index_state, axis],
        outputs=[index_state, volume_state, image, slice_slider, title, mesh, mesh_status, progress],
    )
    correct_button.click(
        lambda *args: mark_label("correct", *args),
        inputs=[files_state, labels_state, timestamps_state, index_state, output_file, axis],
        outputs=[
            labels_state,
            timestamps_state,
            index_state,
            volume_state,
            image,
            slice_slider,
            title,
            mesh,
            mesh_status,
            progress,
        ],
    )
    incorrect_button.click(
        lambda *args: mark_label("incorrect", *args),
        inputs=[files_state, labels_state, timestamps_state, index_state, output_file, axis],
        outputs=[
            labels_state,
            timestamps_state,
            index_state,
            volume_state,
            image,
            slice_slider,
            title,
            mesh,
            mesh_status,
            progress,
        ],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(), server_name="0.0.0.0", server_port=7888)
