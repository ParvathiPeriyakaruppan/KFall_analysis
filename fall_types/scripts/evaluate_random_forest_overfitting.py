"""Audit the saved KFall fall-type Random Forest models for overfitting.

By default, this script evaluates the existing five subject-held-out fold
models. It does not retrain or overwrite those models. Pass --run-extended to
fit separate diagnostic models for learning curves and complexity comparisons.

Run from the repository root:
    python fall_types/scripts/evaluate_random_forest_overfitting.py
    python fall_types/scripts/evaluate_random_forest_overfitting.py --run-extended
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
FALL_TYPES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = FALL_TYPES_DIR.parent
DEFAULT_DATA_DIR = FALL_TYPES_DIR / "results" / "event_aligned"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "random_forest"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "overfitting_evaluation"
TARGETS = ("direction", "context")
KEY_COLUMNS = ("subject", "task_id", "trial_id")
METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=DEFAULT_DATA_DIR / "event_aligned_features.csv",
        help="Trial-level feature CSV created by event_aligned_analysis.py.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing saved fold/final .joblib models and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reports and plots; source models/data are not changed.",
    )
    parser.add_argument(
        "--run-extended",
        action="store_true",
        help=(
            "Also fit separate grouped learning-curve and complexity-test models. "
            "The saved models are never modified."
        ),
    )
    return parser.parse_args()


def metric_values(actual: pd.Series | np.ndarray,
                  predicted: pd.Series | np.ndarray,
                  class_labels: list[str]) -> dict[str, float]:
    """Calculate overall and imbalance-aware metrics with macro averaging."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, labels=class_labels, average="macro", zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, labels=class_labels, average="macro", zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, labels=class_labels, average="macro", zero_division=0
        ),
    }


def print_metric_line(label: str, metrics: dict[str, float]) -> None:
    values = " | ".join(f"{name}={metrics[name]:.4f}" for name in METRIC_NAMES)
    print(f"{label}: {values}")


def mean_and_std(records: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {
        name: (
            float(np.mean([record[name] for record in records])),
            float(np.std([record[name] for record in records])),
        )
        for name in METRIC_NAMES
    }


def read_model_features(model_dir: Path, target: str) -> list[str]:
    feature_file = model_dir / f"{target}_feature_columns.csv"
    if not feature_file.exists():
        raise FileNotFoundError(f"Missing saved feature list: {feature_file}")
    return pd.read_csv(feature_file)["feature"].astype(str).tolist()


def verify_groupkfold_assignments(data: pd.DataFrame, joined: pd.DataFrame,
                                 target: str) -> bool:
    """Confirm each saved fold has the same subject set as GroupKFold."""
    splitter = GroupKFold(n_splits=5)
    expected = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(data, data[target], groups=data["subject"]), 1
    ):
        expected[fold] = set(data.iloc[test_indices]["subject"].unique())
    saved = {
        int(fold): set(rows["subject"].unique())
        for fold, rows in joined.groupby("fold")
    }
    matches = expected == saved
    message = (
        "PASS - saved folds match reconstructed GroupKFold subject groups"
        if matches else
        "WARNING - saved folds differ from reconstructed GroupKFold groups; "
        "checking subject-disjointness independently"
    )
    print(f"GroupKFold assignment check: {message}")
    return matches


def check_feature_leakage(feature_columns: list[str]) -> tuple[str, list[str]]:
    """Flag ID/label-like columns and known event-time/annotation risks."""
    direct_tokens = ("subject", "task_id", "trial_id", "description", "label", "target")
    direct_risks = [
        name for name in feature_columns
        if any(token in name.lower() for token in direct_tokens)
    ]
    annotation_risks = [
        name for name in feature_columns
        if name.lower() in {"onset_to_impact_s", "fall_duration", "event_duration"}
    ]
    post_event_risks = [
        name for name in feature_columns
        if "peak_time" in name.lower()
    ]

    reasons = []
    if direct_risks:
        reasons.append(f"ID/label-like model inputs found: {direct_risks}")
    if annotation_risks:
        reasons.append(
            "features derived from labeled onset/impact annotations: "
            f"{annotation_risks}"
        )
    if post_event_risks:
        reasons.append(
            "peak timing features use an event-aligned window that includes "
            f"post-impact samples: {post_event_risks}"
        )

    if reasons:
        reasons.append(
            "These are temporal/annotation risks for pre-impact deployment; they "
            "do not by themselves prove that the direction/context target was copied."
        )
        return "YES", reasons
    return "NO", ["No obvious ID, target, annotation-duration, or peak-time feature names found."]


def check_preprocessing(model: object) -> tuple[str, str]:
    """Inspect the serialized estimator for a fitted preprocessing pipeline."""
    steps = getattr(model, "steps", None)
    if steps is None:
        return (
            "NO",
            "The saved estimator is a direct RandomForestClassifier; no fitted "
            "scaler or normalization step is serialized with it. The feature "
            "generator uses per-trial summaries rather than dataset-fitted scaling.",
        )

    transformer_steps = [
        name for name, transformer in steps[:-1]
        if any(token in type(transformer).__name__.lower()
               for token in ("scaler", "normalizer", "transformer"))
    ]
    if transformer_steps:
        return (
            "UNKNOWN",
            "A preprocessing step is present in the saved pipeline "
            f"({transformer_steps}), but the serialized estimator alone cannot "
            "prove which subjects were used when it was fitted.",
        )
    return "NO", "No scaler or normalization step was found in the saved pipeline."


def plot_confusion(actual: np.ndarray, predicted: np.ndarray,
                   class_labels: list[str], target: str, output_dir: Path) -> None:
    matrix = confusion_matrix(actual, predicted, labels=class_labels)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"Unseen-subject out-of-fold confusion matrix: {target}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    midpoint = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > midpoint else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_matrix_{target}.png", dpi=180)
    plt.close(figure)


def evaluate_target(features: pd.DataFrame, oof_predictions: pd.DataFrame,
                    target: str, model_dir: Path, output_dir: Path) -> dict:
    feature_columns = read_model_features(model_dir, target)
    missing_columns = [name for name in feature_columns if name not in features.columns]
    if missing_columns:
        raise ValueError(f"{target}: feature table is missing model inputs: {missing_columns}")

    data = features.dropna(subset=feature_columns + [target]).copy()
    if data.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Feature table has duplicate subject/task/trial keys.")

    target_oof = oof_predictions.loc[
        oof_predictions["group"] == target,
        [*KEY_COLUMNS, "fold", "actual", "prediction"],
    ].copy()
    if target_oof.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{target}: OOF predictions have duplicate trial keys.")

    joined = data.merge(target_oof, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if joined["fold"].isna().any():
        missing_count = int(joined["fold"].isna().sum())
        raise ValueError(f"{target}: {missing_count} feature rows have no saved OOF fold.")
    if len(joined) != len(target_oof):
        raise ValueError(
            f"{target}: saved OOF predictions ({len(target_oof)}) and usable "
            f"feature rows ({len(joined)}) do not match."
        )
    if not np.array_equal(joined[target].astype(str), joined["actual"].astype(str)):
        raise ValueError(f"{target}: OOF actual labels disagree with the feature table.")

    fold_count_by_subject = joined.groupby("subject")["fold"].nunique()
    if (fold_count_by_subject != 1).any():
        bad_subjects = fold_count_by_subject[fold_count_by_subject != 1].index.tolist()
        raise ValueError(f"{target}: subjects assigned to multiple folds: {bad_subjects}")
    groupkfold_matches = verify_groupkfold_assignments(data, joined, target)

    class_labels = sorted(data[target].astype(str).unique().tolist())
    fold_records = []
    out_of_fold_actual = []
    out_of_fold_predicted = []
    out_of_fold_rows = []
    assignment_rows = []

    print(f"\n{'=' * 78}\nTARGET: {target.upper()}\n{'=' * 78}")
    print(f"Usable trials: {len(data)} | Subjects: {data['subject'].nunique()}")
    print(f"Model features: {len(feature_columns)}")

    all_subjects = set(data["subject"].unique())
    for fold in sorted(joined["fold"].astype(int).unique()):
        test_rows = joined.loc[joined["fold"].astype(int) == fold]
        test_subjects = set(test_rows["subject"].unique())
        train_subjects = all_subjects - test_subjects
        overlap = train_subjects & test_subjects
        train_rows = data.loc[data["subject"].isin(train_subjects)]
        train_subject_list = [
            subject.item() if isinstance(subject, np.generic) else subject
            for subject in sorted(train_subjects)
        ]
        test_subject_list = [
            subject.item() if isinstance(subject, np.generic) else subject
            for subject in sorted(test_subjects)
        ]
        model_path = model_dir / f"{target}_fold{fold}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing saved fold model: {model_path}")
        model = joblib.load(model_path)

        expected_features = list(getattr(model, "feature_names_in_", feature_columns))
        if expected_features != feature_columns:
            raise ValueError(f"{model_path.name}: saved model feature order does not match CSV.")

        train_prediction = model.predict(train_rows[feature_columns])
        test_prediction = model.predict(test_rows[feature_columns])
        stored_prediction = test_rows["prediction"].astype(str).to_numpy()
        if not np.array_equal(test_prediction.astype(str), stored_prediction):
            raise ValueError(
                f"{model_path.name}: model predictions do not match saved OOF predictions."
            )

        train_metrics = metric_values(
            train_rows[target].astype(str), train_prediction.astype(str), class_labels
        )
        test_metrics = metric_values(
            test_rows["actual"].astype(str), test_prediction.astype(str), class_labels
        )
        passed = not overlap
        print(f"\nFold {fold}: {'PASS - No subject overlap' if passed else 'WARNING - Subject leakage detected'}")
        print(f"Training subject IDs: {train_subject_list}")
        print(f"Testing subject IDs:  {test_subject_list}")
        print_metric_line("Training", train_metrics)
        print_metric_line("Held-out", test_metrics)

        fold_record = {
            "group": target,
            "fold": fold,
            "train_subject_count": len(train_subjects),
            "test_subject_count": len(test_subjects),
            "train_subject_ids": ";".join(map(str, train_subject_list)),
            "test_subject_ids": ";".join(map(str, test_subject_list)),
            "subject_overlap": ";".join(map(str, sorted(overlap))),
        }
        fold_record.update({f"train_{name}": value for name, value in train_metrics.items()})
        fold_record.update({f"heldout_{name}": value for name, value in test_metrics.items()})
        fold_records.append(fold_record)
        assignment_rows.extend(
            {"group": target, "fold": fold, "subject": subject}
            for subject in sorted(test_subjects)
        )
        out_of_fold_actual.extend(test_rows["actual"].astype(str).tolist())
        out_of_fold_predicted.extend(test_prediction.astype(str).tolist())
        fold_output = test_rows[[*KEY_COLUMNS, "actual", "fold"]].copy()
        fold_output["prediction"] = test_prediction
        fold_output["correct"] = fold_output["actual"].astype(str) == fold_output["prediction"].astype(str)
        fold_output["group"] = target
        out_of_fold_rows.append(fold_output)

    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_csv(output_dir / f"fold_metrics_{target}.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(output_dir / f"subject_folds_{target}.csv", index=False)
    prediction_frame = pd.concat(out_of_fold_rows, ignore_index=True)
    prediction_frame.to_csv(output_dir / f"out_of_fold_predictions_{target}.csv", index=False)

    train_summary = mean_and_std([
        {name: float(record[f"train_{name}"]) for name in METRIC_NAMES}
        for record in fold_records
    ])
    heldout_summary = mean_and_std([
        {name: float(record[f"heldout_{name}"]) for name in METRIC_NAMES}
        for record in fold_records
    ])
    pooled_metrics = metric_values(out_of_fold_actual, out_of_fold_predicted, class_labels)
    no_overlap = all(not record["subject_overlap"] for record in fold_records)

    print("\nFive-fold mean +/- standard deviation:")
    for name in METRIC_NAMES:
        train_mean, train_std = train_summary[name]
        test_mean, test_std = heldout_summary[name]
        print(f"{name}: train {train_mean:.4f} +/- {train_std:.4f}; "
              f"held-out {test_mean:.4f} +/- {test_std:.4f}")
    print_metric_line("Pooled out-of-fold", pooled_metrics)
    print("Subject split check: PASS - No subject overlap" if no_overlap
          else "Subject split check: WARNING - Subject leakage detected")

    report = classification_report(
        out_of_fold_actual,
        out_of_fold_predicted,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for label in class_labels:
        class_report = report[label]
        per_class_rows.append({
            "group": target,
            "class": label,
            "precision": class_report["precision"],
            "recall": class_report["recall"],
            "f1": class_report["f1-score"],
            "support": int(class_report["support"]),
        })
    pd.DataFrame(per_class_rows).to_csv(output_dir / f"per_class_metrics_{target}.csv", index=False)
    plot_confusion(
        np.asarray(out_of_fold_actual), np.asarray(out_of_fold_predicted),
        class_labels, target, output_dir,
    )

    final_model_path = model_dir / f"{target}_final.joblib"
    final_model_exists = final_model_path.exists()
    if final_model_exists:
        final_model = joblib.load(final_model_path)
        final_fit_subjects = data["subject"].nunique()
        print(
            f"Final model found and loaded ({type(final_model).__name__}): "
            f"{final_model_path.name}. It is the all-data model "
            f"({final_fit_subjects} subjects), so its own score is not an unseen-subject test."
        )

    return {
        "target": target,
        "data": data,
        "joined": joined,
        "feature_columns": feature_columns,
        "class_labels": class_labels,
        "fold_records": fold_records,
        "train_summary": train_summary,
        "heldout_summary": heldout_summary,
        "pooled_metrics": pooled_metrics,
        "no_overlap": no_overlap,
        "groupkfold_matches": groupkfold_matches,
        "fold_assignments": pd.DataFrame(assignment_rows),
    }


def run_learning_curve(result: dict, output_dir: Path) -> pd.DataFrame:
    """Train diagnostic forests on increasing numbers of training subjects."""
    target = result["target"]
    data = result["data"]
    joined = result["joined"]
    feature_columns = result["feature_columns"]
    class_labels = result["class_labels"]
    fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
    rows = []

    for fold in sorted(joined["fold"].astype(int).unique()):
        test_rows = joined.loc[joined["fold"].astype(int) == fold]
        test_subjects = set(test_rows["subject"].unique())
        train_rows = data.loc[~data["subject"].isin(test_subjects)]
        subject_ids = np.asarray(sorted(train_rows["subject"].unique()))
        for fraction_index, fraction in enumerate(fractions):
            sample_count = max(1, math.ceil(len(subject_ids) * fraction))
            random = np.random.default_rng(4200 + fold * 10 + fraction_index)
            selected_subjects = set(random.choice(subject_ids, size=sample_count, replace=False).tolist())
            subset = train_rows.loc[train_rows["subject"].isin(selected_subjects)]
            diagnostic_model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            diagnostic_model.fit(subset[feature_columns], subset[target])
            train_metrics = metric_values(
                subset[target].astype(str),
                diagnostic_model.predict(subset[feature_columns]).astype(str),
                class_labels,
            )
            validation_metrics = metric_values(
                test_rows[target].astype(str),
                diagnostic_model.predict(test_rows[feature_columns]).astype(str),
                class_labels,
            )
            rows.append({
                "group": target,
                "fold": fold,
                "fraction_of_training_subjects": fraction,
                "training_subject_count": sample_count,
                "training_trial_count": len(subset),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            })

    curve = pd.DataFrame(rows)
    curve.to_csv(output_dir / f"learning_curve_{target}.csv", index=False)
    averaged = curve.groupby("fraction_of_training_subjects").agg(
        train_mean=("train_balanced_accuracy", "mean"),
        train_std=("train_balanced_accuracy", "std"),
        validation_mean=("validation_balanced_accuracy", "mean"),
        validation_std=("validation_balanced_accuracy", "std"),
        training_trials=("training_trial_count", "mean"),
    ).reset_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(averaged["fraction_of_training_subjects"], averaged["train_mean"],
             marker="o", label="Training balanced accuracy")
    axis.plot(averaged["fraction_of_training_subjects"], averaged["validation_mean"],
             marker="o", label="Held-out-subject balanced accuracy")
    axis.set(
        xlabel="Fraction of available training subjects",
        ylabel="Balanced accuracy",
        ylim=(0, 1.02),
        title=f"Grouped learning curve: {target}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"learning_curve_{target}.png", dpi=180)
    plt.close(figure)
    return curve


def complexity_configurations() -> list[tuple[str, dict[str, object]]]:
    """Create one-factor-at-a-time complexity settings around the saved model."""
    baseline = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }
    configurations = [("baseline", baseline.copy())]
    values = {
        "n_estimators": (100, 500),
        "max_depth": (5, 10),
        "min_samples_split": (5, 10),
        "min_samples_leaf": (2, 4),
    }
    for parameter, options in values.items():
        for value in options:
            config = baseline.copy()
            config[parameter] = value
            configurations.append((f"{parameter}={value}", config))
    return configurations


def run_complexity_test(result: dict, output_dir: Path) -> pd.DataFrame:
    """Compare training/validation scores while changing one RF parameter."""
    target = result["target"]
    data = result["data"]
    joined = result["joined"]
    feature_columns = result["feature_columns"]
    class_labels = result["class_labels"]
    configurations = complexity_configurations()
    rows = []

    for configuration_name, parameters in configurations:
        for fold in sorted(joined["fold"].astype(int).unique()):
            test_rows = joined.loc[joined["fold"].astype(int) == fold]
            test_subjects = set(test_rows["subject"].unique())
            train_rows = data.loc[~data["subject"].isin(test_subjects)]
            diagnostic_model = RandomForestClassifier(
                **parameters,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            diagnostic_model.fit(train_rows[feature_columns], train_rows[target])
            train_metrics = metric_values(
                train_rows[target].astype(str),
                diagnostic_model.predict(train_rows[feature_columns]).astype(str),
                class_labels,
            )
            validation_metrics = metric_values(
                test_rows[target].astype(str),
                diagnostic_model.predict(test_rows[feature_columns]).astype(str),
                class_labels,
            )
            rows.append({
                "group": target,
                "configuration": configuration_name,
                **parameters,
                "fold": fold,
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            })

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"complexity_metrics_{target}.csv", index=False)
    averaged = results.groupby("configuration").agg(
        train_mean=("train_balanced_accuracy", "mean"),
        validation_mean=("validation_balanced_accuracy", "mean"),
    ).reset_index()
    positions = np.arange(len(averaged))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions - width / 2, averaged["train_mean"], width, label="Training")
    axis.bar(positions + width / 2, averaged["validation_mean"], width, label="Held-out subjects")
    axis.set(
        xticks=positions,
        xticklabels=averaged["configuration"],
        ylabel="Balanced accuracy",
        ylim=(0, 1.02),
        title=f"Random Forest complexity comparison: {target}",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"complexity_test_{target}.png", dpi=180)
    plt.close(figure)
    return results


def interpretation_for(gap: float, heldout_balanced_accuracy: float) -> str:
    if gap >= 0.10:
        return (
            "Evidence consistent with overfitting: mean in-fold training balanced "
            "accuracy exceeds held-out-subject performance by at least 0.10. "
            "Consider the feature-leakage warning and extended diagnostics too."
        )
    if gap >= 0.05:
        return (
            "Some overfitting is possible: there is a noticeable training-to-held-out "
            "gap, but it is not decisive by itself."
        )
    if heldout_balanced_accuracy >= 0.80:
        return (
            "Evidence of good subject-level generalization in grouped cross-validation: "
            "the training-to-held-out gap is small and held-out balanced accuracy is strong."
        )
    return (
        "Inconclusive: there is no large training-to-held-out gap, but held-out "
        "performance is modest. The model may need better features or more data."
    )


def write_report(results: list[dict], feature_leakage: str,
                 feature_reasons: list[str], preprocessing_status: str,
                 preprocessing_note: str, extended: bool, output_dir: Path) -> None:
    lines = [
        "RANDOM FOREST OVERFITTING REPORT",
        "Evaluation uses saved grouped-fold models and out-of-fold predictions.",
        "There is no separate fixed external test set in the saved artifacts; "
        "held-out-subject scores below are five-fold cross-validation estimates.",
        "Training scores are in-fold resubstitution scores averaged across the "
        "five fold models, not scores from the all-data final model.",
        "",
    ]
    for result in results:
        target = result["target"]
        train_summary = result["train_summary"]
        heldout_summary = result["heldout_summary"]
        gap = (
            train_summary["balanced_accuracy"][0]
            - heldout_summary["balanced_accuracy"][0]
        )
        lines.extend([
            f"TARGET: {target.upper()}",
            "Training Performance (mean +/- standard deviation across fold models):",
        ])
        for name in METRIC_NAMES:
            mean, deviation = train_summary[name]
            lines.append(f"  {name.replace('_', ' ').title()}: {mean:.4f} +/- {deviation:.4f}")
        lines.append("Unseen-Subject Performance (held-out folds):")
        for name in METRIC_NAMES:
            mean, deviation = heldout_summary[name]
            lines.append(f"  {name.replace('_', ' ').title()}: {mean:.4f} +/- {deviation:.4f}")
        lines.extend([
            f"Performance Gap (training minus held-out balanced accuracy): {gap:+.4f}",
            "GroupKFold Assignment Check: " + (
                "PASS - saved folds match reconstructed GroupKFold"
                if result["groupkfold_matches"] else
                "WARNING - see subject-disjointness and saved-split notes"
            ),
            "Subject Leakage: " + (
                "PASS - No subject overlap" if result["no_overlap"]
                else "WARNING - Subject leakage detected"
            ),
            "Cross-Validation Mean Balanced Accuracy: "
            f"{heldout_summary['balanced_accuracy'][0]:.4f}",
            "Cross-Validation Balanced Accuracy Standard Deviation: "
            f"{heldout_summary['balanced_accuracy'][1]:.4f}",
            f"Potential Feature Leakage: {feature_leakage}",
            f"Preprocessing Leakage: {preprocessing_status}",
            "Overall Interpretation: " + interpretation_for(
                gap, heldout_summary["balanced_accuracy"][0]
            ),
            "",
        ])
    lines.extend([
        "LEAKAGE REVIEW",
        *[f"  - {reason}" for reason in feature_reasons],
        f"  - Preprocessing: {preprocessing_note}",
        "",
        "EXTENDED DIAGNOSTICS",
        "  Learning curves and complexity comparisons: "
        + ("completed" if extended else "not run; use --run-extended to run them"),
        "  The optional diagnostics fit temporary models only and do not overwrite "
        "saved .joblib files.",
        "",
        "PLAIN-LANGUAGE SUMMARY",
        "  The training score shows how well each fold model predicts subjects it "
        "learned from. The held-out score shows performance on different subjects. "
        "A large gap is a warning sign, while a small gap with strong held-out "
        "performance supports generalization. Event-aligned post-impact features "
        "are not suitable evidence of pre-impact prediction ability.",
    ])
    report_path = output_dir / "overfitting_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.feature_table.exists():
        raise FileNotFoundError(f"Feature table not found: {args.feature_table}")
    oof_path = args.model_dir / "out_of_fold_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Saved out-of-fold predictions not found: {oof_path}")

    features = pd.read_csv(args.feature_table)
    oof_predictions = pd.read_csv(oof_path)
    required_metadata = {*KEY_COLUMNS, *TARGETS}
    missing_metadata = sorted(required_metadata - set(features.columns))
    if missing_metadata:
        raise ValueError(f"Feature table is missing required columns: {missing_metadata}")
    required_oof = {*KEY_COLUMNS, "actual", "prediction", "fold", "group"}
    missing_oof = sorted(required_oof - set(oof_predictions.columns))
    if missing_oof:
        raise ValueError(f"OOF file is missing required columns: {missing_oof}")

    results = [
        evaluate_target(features, oof_predictions, target, args.model_dir, args.output_dir)
        for target in TARGETS
    ]
    reference_model = joblib.load(args.model_dir / "direction_fold1.joblib")
    feature_leakage, feature_reasons = check_feature_leakage(
        results[0]["feature_columns"]
    )
    preprocessing_status, preprocessing_note = check_preprocessing(reference_model)
    print(f"\nPotential feature leakage: {feature_leakage}")
    for reason in feature_reasons:
        print(f"  - {reason}")
    print(f"Preprocessing leakage: {preprocessing_status}")
    print(f"  {preprocessing_note}")

    if args.run_extended:
        for result in results:
            print(f"\nRunning extended diagnostics for {result['target']}...")
            run_learning_curve(result, args.output_dir)
            run_complexity_test(result, args.output_dir)

    write_report(
        results,
        feature_leakage,
        feature_reasons,
        preprocessing_status,
        preprocessing_note,
        args.run_extended,
        args.output_dir,
    )


if __name__ == "__main__":
    main()