"""Plot KFall sensor signals grouped by fall direction and situation.

The KFall label files identify trials by task and provide a task description.
Direction and context are therefore task-level metadata, not extra labels in the
sensor CSV files. This script preserves the original description and records the
keyword-based grouping in ``fall_type_metadata.csv``.

Run from any directory after extracting ``sensor_data`` and ``label_data``:

    python fall_types/plot_fall_types.py

Outputs are written only below ``fall_types/plots``. Each task, direction group,
and context group gets a multi-panel PNG containing Acc X/Y/Z, Gyr X/Y/Z, and
Euler X/Y/Z versus elapsed time. Thin lines are individual trials and the bold
line is their mean after interpolation onto the shortest common time grid.
"""

import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_TYPES_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(FALL_TYPES_DIR)
PLOTS_DIR = os.path.join(FALL_TYPES_DIR, "results", "plots")
METADATA_CSV = os.path.join(FALL_TYPES_DIR, "results", "metadata", "fall_type_metadata.csv")
MAX_TRIAL_OVERLAYS = 40

PIPELINE_DIR = os.path.join(
    PROJECT_ROOT,
    "fall_pattern_analysis",
    "paper_threshold_validation",
)
sys.path.insert(0, PIPELINE_DIR)
import analyze_pattern as pipeline  # noqa: E402
from analyze_pattern import (  # noqa: E402
    FALL_TASK_IDS,
    discover_subjects,
    load_labels,
    load_sensor_data,
)

DEFAULT_CACHE_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
    "usmanabbasi2002", "kfall-dataset", "versions", "2", "KFall Dataset",
    "KFall Dataset",
)
DATASET_ROOT = os.environ.get("KFALL_DATASET_ROOT", "")
if not DATASET_ROOT:
    local_sensor = os.path.join(PROJECT_ROOT, "sensor_data")
    DATASET_ROOT = PROJECT_ROOT if os.path.isdir(local_sensor) else DEFAULT_CACHE_ROOT
pipeline.SENSOR_DIR = os.path.join(DATASET_ROOT, "sensor_data")
pipeline.LABEL_DIR = os.path.join(DATASET_ROOT, "label_data")

CHANNELS = [
    ("AccX", "Acceleration X (g)", "#d1495b"),
    ("AccY", "Acceleration Y (g)", "#00798c"),
    ("AccZ", "Acceleration Z (g)", "#edae49"),
    ("GyrX", "Gyroscope X (deg/s)", "#6a4c93"),
    ("GyrY", "Gyroscope Y (deg/s)", "#2a9d8f"),
    ("GyrZ", "Gyroscope Z (deg/s)", "#f77f00"),
    ("EulerX", "Euler X / roll (deg)", "#264653"),
    ("EulerY", "Euler Y / pitch (deg)", "#e76f51"),
    ("EulerZ", "Euler Z / yaw (deg)", "#457b9d"),
]

# Descriptions in different KFall releases vary slightly. These are only a
# fallback when a workbook has no usable description for a task.
TASK_FALLBACKS = {
    20: "forward fall while sitting down",
    21: "backward fall while sitting down",
    22: "lateral fall while sitting down",
    23: "forward fall while standing",
    24: "backward fall while standing",
    25: "forward sitting fainting",
    26: "backward sitting fainting",
    27: "lateral sitting fainting",
    28: "forward fall while walking",
    29: "backward fall while walking",
    30: "lateral fall while walking",
    31: "forward fall while running",
    32: "backward fall while running",
    33: "lateral fall while running",
    34: "fall while going downstairs",
}


def classify_description(description):
    """Return direction and situation labels without pretending they are labels."""
    text = str(description).lower()
    if any(word in text for word in ("backward", "backwards", "back fall")):
        direction = "backward"
    elif any(word in text for word in ("lateral", "sideways", "sideway", "side fall")):
        direction = "lateral"
    elif any(word in text for word in ("forward", "front fall")):
        direction = "forward"
    else:
        direction = "unspecified"

    if any(word in text for word in ("faint", "syncope")):
        context = "fainting"
    elif any(word in text for word in ("stair", "stairs")):
        context = "stairs"
    elif any(word in text for word in ("run", "running", "jog")):
        context = "running"
    elif any(word in text for word in ("walk", "walking")):
        context = "walking"
    elif any(word in text for word in ("sit", "sitting", "chair")):
        context = "sitting"
    elif any(word in text for word in ("stand", "standing", "get up")):
        context = "standing"
    elif any(word in text for word in ("lie", "lying", "bed")):
        context = "lying"
    else:
        context = "unspecified"
    return direction, context


def task_description(label_frame, task_id):
    if label_frame is not None:
        matches = label_frame[label_frame["Task Code (Task ID)"].astype(str).str.contains(
            f"({task_id})", regex=False, na=False
        )]
        if not matches.empty:
            values = matches["Description"].dropna().astype(str).str.strip()
            values = values[values.ne("")]
            if not values.empty:
                return values.iloc[0]
    return TASK_FALLBACKS.get(task_id, f"fall task {task_id}")


def collect_trials():
    records = []
    for subject in discover_subjects():
        labels = load_labels(subject)
        for task_id in FALL_TASK_IDS:
            description = task_description(labels, task_id)
            direction, context = classify_description(description)
            for trial_id in range(1, 6):
                frame = load_sensor_data(subject, task_id, trial_id)
                if frame is None or "TimeStamp(s)" not in frame:
                    continue
                if not all(channel in frame for channel, _, _ in CHANNELS):
                    continue
                time = frame["TimeStamp(s)"].to_numpy(float)
                time = time - time[0]
                records.append({
                    "subject": subject,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "description": description,
                    "direction": direction,
                    "context": context,
                    "time": time,
                    "signals": {channel: frame[channel].to_numpy(float)
                                for channel, _, _ in CHANNELS},
                })
    return records


def plot_group(records, group_name, group_value, output_path):
    selected = [record for record in records if record[group_name] == group_value]
    if not selected:
        return
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False)
    for axis, (channel, ylabel, color) in zip(axes.flat, CHANNELS):
        durations = [record["time"][-1] for record in selected]
        common_time = np.linspace(0, min(durations), 300)
        aligned = [np.interp(common_time, record["time"], record["signals"][channel])
                   for record in selected]
        values = np.asarray(aligned)
        overlay_values = values[::max(1, len(values) // MAX_TRIAL_OVERLAYS)]
        for trace in overlay_values[:MAX_TRIAL_OVERLAYS]:
            axis.plot(common_time, trace, color=color, alpha=0.12, linewidth=0.6)
        axis.plot(common_time, values.mean(axis=0), color=color, linewidth=2.0)
        axis.set_title(ylabel)
        axis.set_xlabel("Elapsed time (s)")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.2)
    title = f"{group_name.title()}: {group_value} | n={len(selected)} trials"
    descriptions = sorted({record["description"] for record in selected})
    if len(descriptions) == 1:
        title += f"\n{descriptions[0]}"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    if not os.path.isdir(pipeline.SENSOR_DIR) or not os.path.isdir(pipeline.LABEL_DIR):
        raise SystemExit(
            "Could not find both sensor_data and label_data. Set "
            "KFALL_DATASET_ROOT to their parent directory, or extract them into "
            f"{PROJECT_ROOT}. Checked: {DATASET_ROOT}"
        )
    records = collect_trials()
    if not records:
        raise SystemExit(
            "No fall trials found. Extract sensor_data/ and label_data/ into "
            f"{PROJECT_ROOT} before running this script."
        )

    metadata = pd.DataFrame([
        {key: record[key] for key in
         ("subject", "task_id", "trial_id", "description", "direction", "context")}
        for record in records
    ])
    metadata.to_csv(METADATA_CSV, index=False)

    for task_id in sorted({record["task_id"] for record in records}):
        plot_group(records, "task_id", task_id,
                   os.path.join(PLOTS_DIR, "by_task", f"F{task_id - 19:02d}.png"))
    for group_name in ("direction", "context"):
        for group_value in sorted({record[group_name] for record in records}):
            safe_value = re.sub(r"[^a-z0-9]+", "_", group_value.lower()).strip("_")
            plot_group(records, group_name, group_value,
                       os.path.join(PLOTS_DIR, group_name, f"{safe_value}.png"))
    print(f"Plotted {len(records)} trials; metadata: {METADATA_CSV}; plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()