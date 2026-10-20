"""Compare subject-grouped ConvLSTM and 1D CNN+GRU fall-type classifiers.

Both models consume the same impact-aligned nine-channel sensor sequences and
predict direction and activity context with separate classification heads.
Run from the repository root with:

    python fall_types/scripts/compare_convlstm_cnn_gru.py

Five-fold GroupKFold keeps subjects separate. Normalization and class weights
are fit within each training fold. Out-of-fold results and a Markdown report
are written below fall_types/results/event_aligned/recurrent_models/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
FALL_TYPES_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from event_aligned_analysis import EVENT_CHANNELS, collect_event_trials  # noqa: E402

OUTPUT_DIR = FALL_TYPES_DIR / "results" / "event_aligned" / "recurrent_models"
SEED = 42


class ConvRecurrentClassifier(nn.Module):
    """1D convolutional sequence encoder followed by LSTM or GRU and two heads."""

    def __init__(self, channels, direction_classes, context_classes, recurrent="lstm"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
        recurrent_type = nn.LSTM if recurrent == "lstm" else nn.GRU
        self.recurrent = recurrent_type(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.direction_head = nn.Linear(64, direction_classes)
        self.context_head = nn.Linear(64, context_classes)

    def forward(self, inputs):
        sequence = self.conv(inputs).transpose(1, 2)
        recurrent_output, _ = self.recurrent(sequence)
        features = self.dropout(recurrent_output[:, -1, :])
        return self.direction_head(features), self.context_head(features)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(values):
    classes = sorted(pd.unique(values).tolist())
    mapping = {label: i for i, label in enumerate(classes)}
    return np.asarray([mapping[value] for value in values], dtype=np.int64), classes


def class_weights(labels, n_classes, device):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("A training fold is missing a target class; reduce folds or check labels.")
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32, device=device)


def normalize(train, test):
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train.std(axis=(0, 2), keepdims=True), 1e-6)
    return (train - mean) / std, (test - mean) / std, mean.reshape(-1), std.reshape(-1)


def fit(model, x, y_direction, y_context, args, device):
    data = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y_direction), torch.from_numpy(y_context))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    loss_direction = nn.CrossEntropyLoss(weight=class_weights(y_direction, model.direction_head.out_features, device))
    loss_context = nn.CrossEntropyLoss(weight=class_weights(y_context, model.context_head.out_features, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        model.train()
        for bx, by_direction, by_context in loader:
            bx = bx.to(device)
            logits_direction, logits_context = model(bx)
            loss = loss_direction(logits_direction, by_direction.to(device))
            loss = loss + args.context_loss_weight * loss_context(logits_context, by_context.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    outputs = model(torch.from_numpy(x).float().to(device))
    return [torch.softmax(output, dim=1).cpu().numpy() for output in outputs]


def build_inputs(records):
    x = np.asarray([np.stack([r["aligned"][channel] for channel in EVENT_CHANNELS]) for r in records], dtype=np.float32)
    return x, pd.DataFrame([r["meta"] for r in records])


def metric_row(model_name, fold, target, truth, predicted, groups):
    return {
        "model": model_name, "fold": fold, "target": target,
        "held_out_subjects": ";".join(sorted(set(map(str, groups)))),
        "trials": len(truth),
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
    }


def write_report(folds, metadata, args, device):
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    summary = folds.groupby(["model", "target"])[metrics].agg(["mean", "std"])
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv")
    lines = [
        "# ConvLSTM vs 1D CNN+GRU fall-type evaluation", "",
        "## Evaluation setup", "",
        f"- Data: {len(metadata)} impact-aligned fall trials from {metadata.subject.nunique()} subjects.",
        f"- Input: {len(EVENT_CHANNELS)} sensor channels over 201 samples (-1 to +1 second around impact).",
        f"- Validation: {args.folds}-fold GroupKFold by subject; each subject is held out from training in its fold.",
        "- Preprocessing: per-channel mean and standard deviation fit on training subjects in each fold.",
        "- Targets: task-description-derived direction and context; these are not independently annotated trial labels.",
        f"- Training: {args.epochs} epochs per fold, batch size {args.batch_size}, seed {args.seed}, device `{device}`.",
        "- Metrics: accuracy, balanced accuracy (primary under class imbalance), and macro F1. Reported values are mean ± standard deviation across folds.", "",
        "## Results", "",
        "| Model | Target | Accuracy | Balanced accuracy | Macro F1 |", "|---|---|---:|---:|---:|",
    ]
    for model in ("ConvLSTM", "CNN+GRU"):
        for target in ("direction", "context"):
            row = summary.loc[(model, target)]
            cells = [f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}" for m in metrics]
            lines.append(f"| {model} | {target} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Paired fold differences", "",
                  "Difference is CNN+GRU minus ConvLSTM, computed on the same held-out subject folds. Positive values favor CNN+GRU.",
                  "| Target | Metric | Mean difference | Fold SD |", "|---|---|---:|---:|"])
    for target in ("direction", "context"):
        a = folds[(folds.model == "ConvLSTM") & (folds.target == target)].set_index("fold")
        b = folds[(folds.model == "CNN+GRU") & (folds.target == target)].set_index("fold")
        for metric in metrics:
            delta = b[metric] - a[metric]
            lines.append(f"| {target} | {metric.replace('_', ' ')} | {delta.mean():+.3f} | {delta.std(ddof=1):.3f} |")
    lines.extend([
        "", "## Interpretation", "",
        "Use balanced accuracy as the main comparison because class counts may differ. Fold-to-fold standard deviations describe sensitivity to the held-out subject partition; with five folds they are not a substitute for a larger repeated validation study.",
        "The two heads are evaluated separately. Context and direction labels inherit the limitations of the task-description mapping, so these scores should not be interpreted as clinical or real-world performance.", "",
        "Artifacts: `fold_metrics.csv` contains per-fold scores; `out_of_fold_predictions.csv` contains held-out trial predictions; `summary_metrics.csv` contains aggregate scores; `fold_*_*.pt` contains fold model weights and training-fold normalization.",
        "",
    ])
    (OUTPUT_DIR / "accuracy_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(args):
    if args.folds < 2 or args.epochs < 1:
        raise ValueError("folds must be at least 2 and epochs must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    records = collect_event_trials()
    if not records:
        raise SystemExit("No valid event-aligned fall trials found; check the KFall dataset path.")
    inputs, metadata = build_inputs(records)
    direction, direction_classes = encode(metadata.direction)
    context, context_classes = encode(metadata.context)
    groups = metadata.subject.to_numpy()
    if metadata.subject.nunique() < args.folds:
        raise ValueError("Number of folds cannot exceed the number of unique subjects")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_rows, prediction_rows = [], []
    splits = list(GroupKFold(n_splits=args.folds).split(inputs, direction, groups))
    models = (("ConvLSTM", "lstm"), ("CNN+GRU", "gru"))
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train_x, test_x, mean, std = normalize(inputs[train_idx], inputs[test_idx])
        held_groups = groups[test_idx]
        for model_name, recurrent in models:
            set_seed(args.seed + fold)
            model = ConvRecurrentClassifier(inputs.shape[1], len(direction_classes), len(context_classes), recurrent).to(device)
            fit(model, train_x, direction[train_idx], context[train_idx], args, device)
            prob_direction, prob_context = predict(model, test_x, device)
            pred_direction, pred_context = prob_direction.argmax(1), prob_context.argmax(1)
            fold_rows.append(metric_row(model_name, fold, "direction", direction[test_idx], pred_direction, held_groups))
            fold_rows.append(metric_row(model_name, fold, "context", context[test_idx], pred_context, held_groups))
            for local, idx in enumerate(test_idx):
                row = metadata.iloc[idx].to_dict()
                row.update({
                    "model": model_name, "fold": fold,
                    "direction_true": direction_classes[direction[idx]],
                    "direction_predicted": direction_classes[pred_direction[local]],
                    "direction_confidence": float(prob_direction[local].max()),
                    "context_true": context_classes[context[idx]],
                    "context_predicted": context_classes[pred_context[local]],
                    "context_confidence": float(prob_context[local].max()),
                })
                prediction_rows.append(row)
            torch.save({
                "model_state_dict": model.state_dict(), "model": model_name,
                "channels": EVENT_CHANNELS, "direction_classes": direction_classes,
                "context_classes": context_classes, "channel_mean": mean.tolist(),
                "channel_std": std.tolist(), "config": vars(args),
            }, OUTPUT_DIR / f"fold_{fold}_{'convlstm' if recurrent == 'lstm' else 'cnn_gru'}.pt")
            score = fold_rows[-2:]
            print(f"Fold {fold}/{args.folds} {model_name}: direction BA={score[0]['balanced_accuracy']:.3f}, context BA={score[1]['balanced_accuracy']:.3f}", flush=True)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)
    summary = write_report(fold_metrics, metadata, args, device)
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "device": str(device), "channels": EVENT_CHANNELS,
                   "direction_classes": direction_classes, "context_classes": context_classes,
                   "subjects": int(metadata.subject.nunique()), "trials": len(metadata)}, f, indent=2)
    print(f"\nReport: {OUTPUT_DIR / 'accuracy_comparison_report.md'}")
    print(summary.to_string())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--context-loss-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
