"""Event-aligned KFall feature analysis for direction and context.

This script uses the labeled onset and impact frames and aligns every fall to
impact (impact = 0 s) on a -1 s to +1 s grid. It writes all outputs below
``fall_types/event_aligned``.

The feature table contains onset-to-impact duration, acceleration and gyro
magnitude peaks/timing, orientation change from a pre-event baseline, and peak
plus median absolute values for every raw axis. Separability is tested with
Kruskal-Wallis effect-size statistics and subject-grouped random-forest
balanced accuracy.
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_TYPES_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from plot_fall_types import (  # noqa: E402
    CHANNELS,
    FALL_TASK_IDS,
    classify_description,
    discover_subjects,
    load_labels,
    load_sensor_data,
    pipeline,
    task_description,
)

OUTPUT_DIR = os.path.join(FALL_TYPES_DIR, "results", "event_aligned")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "event_aligned_features.csv")
SEPARABILITY_CSV = os.path.join(OUTPUT_DIR, "separability_tests.csv")
CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "grouped_classification.csv")
GRID = np.linspace(-1.0, 1.0, 201)
EVENT_CHANNELS = [channel for channel, _, _ in CHANNELS]


def interpolate_event(time, values, event_time):
    """Interpolate a signal onto the impact-centered, fixed time grid."""
    relative_time = time - event_time
    if relative_time[0] > GRID[0] or relative_time[-1] < GRID[-1]:
        return None
    return np.interp(GRID, relative_time, values)


def feature_values(aligned, duration):
    features = {"onset_to_impact_s": duration}
    acceleration = np.sqrt(
        aligned["AccX"] ** 2 + aligned["AccY"] ** 2 + aligned["AccZ"] ** 2
    )
    gyro = np.sqrt(
        aligned["GyrX"] ** 2 + aligned["GyrY"] ** 2 + aligned["GyrZ"] ** 2
    )
    baseline_mask = (GRID >= -1.0) & (GRID <= -0.5)
    baseline = np.array([
        aligned["EulerX"][baseline_mask].mean(),
        aligned["EulerY"][baseline_mask].mean(),
        aligned["EulerZ"][baseline_mask].mean(),
    ])
    angles = np.column_stack([
        aligned["EulerX"], aligned["EulerY"], aligned["EulerZ"]
    ])
    orientation_change = np.linalg.norm(angles - baseline, axis=1)

    derived = {
        "acc_m": acceleration,
        "gyro_m": gyro,
        "orientation_change": orientation_change,
    }
    for name, values in derived.items():
        peak_index = int(np.argmax(np.abs(values)))
        features[f"{name}_peak"] = float(np.max(np.abs(values)))
        features[f"{name}_peak_time_s"] = float(GRID[peak_index])
        features[f"{name}_median_abs"] = float(np.median(np.abs(values)))

    for channel in EVENT_CHANNELS:
        values = aligned[channel]
        features[f"{channel}_peak_abs"] = float(np.max(np.abs(values)))
        features[f"{channel}_median_abs"] = float(np.median(np.abs(values)))
    return features


def collect_event_trials():
    records = []
    for subject in discover_subjects():
        labels = load_labels(subject)
        for task_id in FALL_TASK_IDS:
            description = task_description(labels, task_id)
            direction, context = classify_description(description)
            for trial_id in range(1, 6):
                frame = load_sensor_data(subject, task_id, trial_id)
                label = pipeline.get_fall_label_info(labels, task_id, trial_id)
                if frame is None or label is None:
                    continue
                if not all(channel in frame for channel in EVENT_CHANNELS):
                    continue
                timestamps = frame["TimeStamp(s)"].to_numpy(float)
                frames = frame["FrameCounter"].to_numpy(float)
                onset, impact = label["onset"], label["impact"]
                if onset < frames.min() or impact > frames.max() or impact <= onset:
                    continue
                onset_time = np.interp(onset, frames, timestamps)
                impact_time = np.interp(impact, frames, timestamps)
                aligned = {}
                for channel in EVENT_CHANNELS:
                    aligned[channel] = interpolate_event(
                        timestamps, frame[channel].to_numpy(float), impact_time
                    )
                if any(values is None for values in aligned.values()):
                    continue
                features = feature_values(aligned, impact_time - onset_time)
                features.update({
                    "subject": subject,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "description": description,
                    "direction": direction,
                    "context": context,
                })
                records.append({"meta": features, "aligned": aligned})
    return records


def plot_event_group(records, group_name, group_value):
    selected = [record for record in records if record["meta"][group_name] == group_value]
    if not selected:
        return
    plot_channels = [
        (channel, title, color) for channel, title, color in CHANNELS
    ] + [
        ("acc_m", "Acceleration magnitude (g)", "#111111"),
        ("gyro_m", "Gyroscope magnitude (deg/s)", "#6a4c93"),
        ("orientation_change", "Orientation change from baseline (deg)", "#e76f51"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True)
    for axis, (channel, title, color) in zip(axes.flat, plot_channels):
        traces = []
        for record in selected:
            aligned = record["aligned"]
            if channel == "acc_m":
                values = np.sqrt(aligned["AccX"] ** 2 + aligned["AccY"] ** 2 + aligned["AccZ"] ** 2)
            elif channel == "gyro_m":
                values = np.sqrt(aligned["GyrX"] ** 2 + aligned["GyrY"] ** 2 + aligned["GyrZ"] ** 2)
            elif channel == "orientation_change":
                baseline_mask = (GRID >= -1.0) & (GRID <= -0.5)
                baseline = np.array([
                    aligned["EulerX"][baseline_mask].mean(),
                    aligned["EulerY"][baseline_mask].mean(),
                    aligned["EulerZ"][baseline_mask].mean(),
                ])
                angles = np.column_stack([aligned["EulerX"], aligned["EulerY"], aligned["EulerZ"]])
                values = np.linalg.norm(angles - baseline, axis=1)
            else:
                values = aligned[channel]
            traces.append(values)
        values = np.asarray(traces)
        mean, std = values.mean(axis=0), values.std(axis=0)
        axis.plot(GRID, mean, color=color, linewidth=2)
        axis.fill_between(GRID, mean - std, mean + std, color=color, alpha=0.18)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Time relative to impact (s)")
        axis.set_ylabel("Value")
    fig.suptitle(f"Impact-aligned signals: {group_name} = {group_value} | n={len(selected)}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = os.path.join(OUTPUT_DIR, group_name)
    os.makedirs(output_dir, exist_ok=True)
    safe_value = re.sub(r"[^a-z0-9]+", "_", str(group_value).lower()).strip("_")
    fig.savefig(os.path.join(output_dir, f"{safe_value}.png"), dpi=180)
    plt.close(fig)


def derived_signal(aligned, channel):
    if channel == "acc_m":
        return np.sqrt(aligned["AccX"] ** 2 + aligned["AccY"] ** 2 + aligned["AccZ"] ** 2)
    if channel == "gyro_m":
        return np.sqrt(aligned["GyrX"] ** 2 + aligned["GyrY"] ** 2 + aligned["GyrZ"] ** 2)
    if channel == "orientation_change":
        baseline_mask = (GRID >= -1.0) & (GRID <= -0.5)
        baseline = np.array([
            aligned["EulerX"][baseline_mask].mean(),
            aligned["EulerY"][baseline_mask].mean(),
            aligned["EulerZ"][baseline_mask].mean(),
        ])
        angles = np.column_stack([aligned["EulerX"], aligned["EulerY"], aligned["EulerZ"]])
        return np.linalg.norm(angles - baseline, axis=1)
    return aligned[channel]


def plot_comparison_groups(records, group_name):
    """Compare all categories in four panels per sensor family."""
    group_values = sorted({record["meta"][group_name] for record in records})
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(group_values)))
    families = {
        "acceleration": [
            ("AccX", "Acceleration X (g)"),
            ("AccY", "Acceleration Y (g)"),
            ("AccZ", "Acceleration Z (g)"),
            ("acc_m", "Acceleration magnitude (g)"),
        ],
        "gyroscope": [
            ("GyrX", "Gyroscope X (deg/s)"),
            ("GyrY", "Gyroscope Y (deg/s)"),
            ("GyrZ", "Gyroscope Z (deg/s)"),
            ("gyro_m", "Gyroscope magnitude (deg/s)"),
        ],
        "orientation": [
            ("EulerX", "Euler X / roll (deg)"),
            ("EulerY", "Euler Y / pitch (deg)"),
            ("EulerZ", "Euler Z / yaw (deg)"),
            ("orientation_change", "Orientation change from baseline (deg)"),
        ],
    }
    output_dir = os.path.join(OUTPUT_DIR, "comparison", group_name)
    os.makedirs(output_dir, exist_ok=True)
    for family_name, channels in families.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        for axis, (channel, title) in zip(axes.flat, channels):
            for group_value, color in zip(group_values, colors):
                selected = [record for record in records
                            if record["meta"][group_name] == group_value]
                traces = np.asarray([
                    derived_signal(record["aligned"], channel) for record in selected
                ])
                axis.plot(GRID, traces.mean(axis=0), color=color, linewidth=2,
                          label=f"{group_value} (n={len(selected)})")
            axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("Time relative to impact (s)")
            axis.set_ylabel("Value")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle(f"Impact-aligned {family_name}: comparison by {group_name}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(output_dir, f"{family_name}.png"), dpi=180)
        plt.close(fig)


def run_separability(features):
    excluded = {"subject", "task_id", "trial_id", "description", "direction", "context"}
    feature_columns = [column for column in features.columns if column not in excluded]
    tests = []
    for group_name in ("direction", "context"):
        grouped = [group for _, group in features.groupby(group_name)]
        for feature in feature_columns:
            samples = [group[feature].dropna().to_numpy() for group in grouped]
            if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
                continue
            statistic, p_value = kruskal(*samples)
            n = sum(len(sample) for sample in samples)
            k = len(samples)
            eta_squared = max(0.0, (statistic * (n - k)) / (n * n - 1))
            tests.append({
                "group": group_name,
                "feature": feature,
                "kruskal_h": statistic,
                "p_value": p_value,
                "epsilon_squared": eta_squared,
            })
    return pd.DataFrame(tests)


def run_grouped_classification(features):
    excluded = {"subject", "task_id", "trial_id", "description", "direction", "context"}
    feature_columns = [column for column in features.columns if column not in excluded]
    rows = []
    for group_name in ("direction", "context"):
        data = features.dropna(subset=feature_columns + [group_name])
        splitter = GroupKFold(n_splits=5)
        classifier = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        )
        scores = cross_val_score(
            classifier, data[feature_columns], data[group_name],
            groups=data["subject"], cv=splitter, scoring="balanced_accuracy",
        )
        rows.append({
            "group": group_name,
            "subjects": data["subject"].nunique(),
            "trials": len(data),
            "balanced_accuracy_mean": scores.mean(),
            "balanced_accuracy_std": scores.std(),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = collect_event_trials()
    if not records:
        raise SystemExit("No valid onset/impact-aligned fall trials found.")
    features = pd.DataFrame([record["meta"] for record in records])
    features.to_csv(FEATURES_CSV, index=False)
    if os.environ.get("KFALL_SKIP_TESTS") != "1":
        separability = run_separability(features)
        separability.to_csv(SEPARABILITY_CSV, index=False)
        run_grouped_classification(features).to_csv(CLASSIFICATION_CSV, index=False)
    for group_name in ("direction", "context"):
        if os.environ.get("KFALL_COMPARISON_ONLY") != "1":
            for group_value in sorted(features[group_name].unique()):
                plot_event_group(records, group_name, group_value)
        plot_comparison_groups(records, group_name)
    print(f"Analyzed {len(records)} trials")
    print(f"Features: {FEATURES_CSV}")
    print(f"Separability tests: {SEPARABILITY_CSV}")
    print(f"Grouped classification: {CLASSIFICATION_CSV}")


if __name__ == "__main__":
    main()