"""Subject-grouped multi-task 1D CNN for KFall direction and context.

The model receives the same impact-aligned -1 to +1 second sensor windows as
the event-aligned Random Forest analysis, but learns directly from the nine raw
accelerometer, gyroscope, and Euler-angle traces.  A shared CNN backbone feeds
two classification heads: one for fall direction and one for activity context.

Run from the repository root::

    python fall_types/scripts/multitask_1d_cnn.py

Results are out-of-fold predictions and balanced accuracies from five-fold
subject-grouped validation.  They are therefore comparable to the Random
Forest results, rather than in-sample training scores.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
FALL_TYPES_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from event_aligned_analysis import EVENT_CHANNELS, collect_event_trials  # noqa: E402


OUTPUT_DIR = FALL_TYPES_DIR / "results" / "event_aligned" / "multitask_1d_cnn"
DEFAULT_SEED = 42


class MultiTaskCNN(nn.Module):
    """Compact temporal CNN with a shared feature extractor and two heads."""

    def __init__(self, channels, direction_classes, context_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(96, 64), nn.ReLU(inplace=True)
        )
        self.direction_head = nn.Linear(64, direction_classes)
        self.context_head = nn.Linear(64, context_classes)

    def forward(self, inputs):
        features = self.embedding(self.backbone(inputs))
        return self.direction_head(features), self.context_head(features)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_labels(values):
    classes = sorted(pd.unique(values).tolist())
    lookup = {name: index for index, name in enumerate(classes)}
    return np.asarray([lookup[value] for value in values], dtype=np.int64), classes


def channel_standardize(train, test):
    """Fit per-channel statistics on a fold's training subjects only."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train - mean) / std, (test - mean) / std, mean.squeeze(), std.squeeze()


def class_weights(labels, class_count, device):
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training fold has no examples for class indices {missing}.")
    weights = len(labels) / (class_count * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_fold(model, train_x, train_direction, train_context, args, device):
    dataset = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_direction),
        torch.from_numpy(train_context),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    direction_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_direction, model.direction_head.out_features, device)
    )
    context_loss = nn.CrossEntropyLoss(
        weight=class_weights(train_context, model.context_head.out_features, device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_direction, batch_context in loader:
            batch_x = batch_x.to(device)
            logits_direction, logits_context = model(batch_x)
            loss = direction_loss(logits_direction, batch_direction.to(device))
            loss = loss + args.context_loss_weight * context_loss(logits_context, batch_context.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict(model, inputs, device):
    model.eval()
    logits_direction, logits_context = model(torch.from_numpy(inputs).float().to(device))
    direction_probability = torch.softmax(logits_direction, dim=1).cpu().numpy()
    context_probability = torch.softmax(logits_context, dim=1).cpu().numpy()
    return direction_probability, context_probability


def build_input(records):
    """Convert aligned channel dictionaries into (trial, channel, time) tensors."""
    inputs = np.asarray([
        np.stack([record["aligned"][channel] for channel in EVENT_CHANNELS])
        for record in records
    ], dtype=np.float32)
    metadata = pd.DataFrame([record["meta"] for record in records])
    return inputs, metadata


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    records = collect_event_trials()
    if not records:
        raise SystemExit("No valid onset/impact-aligned fall trials found.")
    inputs, metadata = build_input(records)
    direction, direction_classes = encode_labels(metadata["direction"])
    context, context_classes = encode_labels(metadata["context"])
    groups = metadata["subject"].to_numpy()
    splitter = GroupKFold(n_splits=args.folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(inputs, direction, groups), start=1):
        train_x, test_x, mean, std = channel_standardize(inputs[train_index], inputs[test_index])
        model = MultiTaskCNN(inputs.shape[1], len(direction_classes), len(context_classes)).to(device)
        train_fold(model, train_x, direction[train_index], context[train_index], args, device)
        direction_probability, context_probability = predict(model, test_x, device)
        predicted_direction = direction_probability.argmax(axis=1)
        predicted_context = context_probability.argmax(axis=1)
        direction_score = balanced_accuracy_score(direction[test_index], predicted_direction)
        context_score = balanced_accuracy_score(context[test_index], predicted_context)
        fold_rows.extend([
            {"fold": fold, "target": "direction", "held_out_subjects": ";".join(map(str, sorted(groups[test_index].astype(str)))), "trials": len(test_index), "balanced_accuracy": direction_score},
            {"fold": fold, "target": "context", "held_out_subjects": ";".join(map(str, sorted(groups[test_index].astype(str)))), "trials": len(test_index), "balanced_accuracy": context_score},
        ])
        for local_index, row_index in enumerate(test_index):
            row = metadata.iloc[row_index].to_dict()
            row.update({
                "fold": fold,
                "direction_true": direction_classes[direction[row_index]],
                "direction_predicted": direction_classes[predicted_direction[local_index]],
                "direction_confidence": float(direction_probability[local_index].max()),
                "context_true": context_classes[context[row_index]],
                "context_predicted": context_classes[predicted_context[local_index]],
                "context_confidence": float(context_probability[local_index].max()),
            })
            prediction_rows.append(row)
        torch.save({
            "model_state_dict": model.state_dict(),
            "channels": EVENT_CHANNELS,
            "direction_classes": direction_classes,
            "context_classes": context_classes,
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "config": vars(args),
        }, OUTPUT_DIR / f"fold_{fold}_model.pt")
        print(f"Fold {fold}/{args.folds}: direction={direction_score:.3f}, context={context_score:.3f}")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)
    summary = (fold_metrics.groupby("target")["balanced_accuracy"].agg(["mean", "std"])
               .reset_index().rename(columns={"mean": "balanced_accuracy_mean", "std": "balanced_accuracy_std"}))
    summary.insert(1, "subjects", metadata["subject"].nunique())
    summary.insert(2, "trials", len(metadata))
    summary.to_csv(OUTPUT_DIR / "grouped_classification.csv", index=False)
    with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as output:
        json.dump({**vars(args), "device": str(device), "channels": EVENT_CHANNELS,
                   "direction_classes": direction_classes, "context_classes": context_classes}, output, indent=2)
    print(f"Saved results to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=75, help="Training epochs per validation fold.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--context-loss-weight", type=float, default=1.0,
                        help="Relative contribution of the context loss.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu", action="store_true", help="Do not use CUDA even when it is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
