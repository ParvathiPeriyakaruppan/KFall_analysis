"""Audit saved 1D CNN, ConvLSTM and CNN+GRU fold models for overfitting.

Run after the CNN and recurrent-model cross-validation scripts have produced
their fold checkpoints. Uses each checkpoint's original subject-grouped split
and training-fold normalization, then compares training and held-out scores.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
FALL_TYPES_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_convlstm_cnn_gru import ConvRecurrentClassifier, build_inputs, encode
from event_aligned_analysis import EVENT_CHANNELS, collect_event_trials
from multitask_1d_cnn import MultiTaskCNN

ROOT = FALL_TYPES_DIR / "results" / "event_aligned"
CNN_DIR = ROOT / "multitask_1d_cnn"
RECURRENT_DIR = ROOT / "recurrent_models"
OUT_DIR = ROOT / "overfitting_audit"


def score_split(model, x, y_direction, y_context, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().to(device))
    rows = []
    for target, truth, output in zip(("direction", "context"), (y_direction, y_context), logits):
        predicted = output.argmax(dim=1).cpu().numpy()
        rows.append({
            "target": target,
            "accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        })
    return rows


def load_checkpoint(path, device):
    # weights_only=False is needed because these project checkpoints also store
    # label lists and run configuration alongside the model tensors.
    return torch.load(path, map_location=device, weights_only=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = collect_event_trials()
    if not records:
        raise SystemExit("No event-aligned trials found; check the KFall dataset path.")
    inputs, metadata = build_inputs(records)
    direction, direction_classes = encode(metadata.direction)
    context, context_classes = encode(metadata.context)
    groups = metadata.subject.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(inputs, direction, groups))
    model_specs = [
        ("1D CNN", CNN_DIR, "fold_{fold}_model.pt", "cnn"),
        ("ConvLSTM", RECURRENT_DIR, "fold_{fold}_convlstm.pt", "lstm"),
        ("CNN+GRU", RECURRENT_DIR, "fold_{fold}_cnn_gru.pt", "gru"),
    ]
    rows = []
    for model_name, folder, filename_pattern, kind in model_specs:
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            checkpoint_path = folder / filename_pattern.format(fold=fold)
            if not checkpoint_path.exists():
                raise SystemExit(f"Missing checkpoint: {checkpoint_path}. Train/evaluate that model first.")
            checkpoint = load_checkpoint(checkpoint_path, device)
            if checkpoint["direction_classes"] != direction_classes or checkpoint["context_classes"] != context_classes:
                raise ValueError(f"Label map mismatch in {checkpoint_path}")
            if kind == "cnn":
                model = MultiTaskCNN(len(EVENT_CHANNELS), len(direction_classes), len(context_classes))
            else:
                recurrent = "lstm" if kind == "lstm" else "gru"
                model = ConvRecurrentClassifier(len(EVENT_CHANNELS), len(direction_classes), len(context_classes), recurrent)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)[None, :, None]
            std = np.maximum(np.asarray(checkpoint["channel_std"], dtype=np.float32)[None, :, None], 1e-6)
            for split_name, indices in (("train", train_idx), ("held-out", test_idx)):
                normalized = (inputs[indices] - mean) / std
                scores = score_split(model, normalized, direction[indices], context[indices], device)
                for row in scores:
                    row.update({"model": model_name, "fold": fold, "split": split_name, "trials": len(indices)})
                    rows.append(row)
            print(f"Audited {model_name} fold {fold}/5", flush=True)

    results = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "fold_train_heldout_metrics.csv", index=False)
    summary = results.groupby(["model", "target", "split"])[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    summary.to_csv(OUT_DIR / "summary_metrics.csv")

    lines = [
        "# Deep-learning overfitting audit", "",
        f"Compared saved five-fold subject-held-out models on {len(metadata)} trials from {metadata.subject.nunique()} subjects.",
        "Training and held-out scores use the same fold model and training-fold normalization. A large train-to-held-out gap can indicate overfitting.",
        "Balanced accuracy is the main metric because class frequencies differ. Values are mean ± standard deviation across folds.", "",
        "| Model | Target | Split | Accuracy | Balanced accuracy | Macro F1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model_name in ("1D CNN", "ConvLSTM", "CNN+GRU"):
        for target in ("direction", "context"):
            for split_name in ("train", "held-out"):
                row = summary.loc[(model_name, target, split_name)]
                cells = [f"{row[(metric, 'mean')]:.3f} ± {row[(metric, 'std')]:.3f}" for metric in ("accuracy", "balanced_accuracy", "macro_f1")]
                lines.append(f"| {model_name} | {target} | {split_name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Train-to-held-out balanced-accuracy gap", "",
                  "Gap = training-fold mean minus held-out-fold mean; larger positive values indicate a wider generalization gap.",
                  "| Model | Target | Gap |", "|---|---|---:|"])
    for model_name in ("1D CNN", "ConvLSTM", "CNN+GRU"):
        for target in ("direction", "context"):
            train_score = summary.loc[(model_name, target, "train"), ("balanced_accuracy", "mean")]
            test_score = summary.loc[(model_name, target, "held-out"), ("balanced_accuracy", "mean")]
            gap = train_score - test_score
            lines.append(f"| {model_name} | {target} | {gap:+.3f} ({gap * 100:+.1f} percentage points) |")
    lines.extend(["", "The gap is a diagnostic rather than a formal significance test. Labels are derived from task descriptions, so the audit measures generalization to unseen participants under this dataset's labeling scheme.", ""])
    report_path = OUT_DIR / "overfitting_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
