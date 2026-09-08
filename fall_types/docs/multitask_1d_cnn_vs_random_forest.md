# Multi-task 1D CNN for fall direction and activity context

`scripts/multitask_1d_cnn.py` is the deep-learning counterpart to the
event-aligned Random Forest analysis. It describes an already detected fall;
it is not the binary fall detector.

## Input and labels

Each fall is aligned to its labeled impact frame on the same `-1` to `+1`
second, 201-sample grid used by `event_aligned_analysis.py`. The CNN takes a
`9 x 201` tensor: AccX/Y/Z, GyrX/Y/Z, and EulerX/Y/Z. It receives raw aligned
signals rather than the Random Forest's hand-engineered peak features.

It has a shared three-layer temporal convolutional backbone, global temporal
pooling, and two independent softmax heads:

```text
9 sensor channels x 201 samples
            |
      shared 1D CNN
         /       \
 direction head   context head
```

The targets remain the existing task-description-derived direction and context
labels. This limitation still applies: the labels are task-level categories,
not independently annotated per-trial ground truth.

The context head uses exactly six classes: `fainting`, `running`, `sitting`,
`standing`, `walking`, and `unspecified`. Task descriptions outside those
classes (such as stairs or lying) map to `unspecified`.

## Evaluation

The script uses 5-fold `GroupKFold` with subject as the group, exactly as the
Random Forest workflow does. Per-channel standardization is fitted only on the
training subjects of each fold. Both losses use class weights computed from
that fold's training trials. Results therefore report performance on unseen
subjects without normalization or subject leakage.

The default command is:

```powershell
python fall_types/scripts/multitask_1d_cnn.py
```

Useful options include `--epochs 100`, `--batch-size 64`, and `--cpu`.
Outputs are written to `results/event_aligned/multitask_1d_cnn/`:

- `grouped_classification.csv` — mean and standard deviation of held-out
  balanced accuracy for each head.
- `fold_metrics.csv` — one direction and context score per held-out fold.
- `out_of_fold_predictions.csv` — trial-level held-out labels, predictions,
  and confidences.
- `fold_*_model.pt` — model weights, fold-specific normalizer, label maps, and
  run configuration.

The defaults deliberately keep the CNN compact for the roughly 2,315 trials.
Compare its held-out balanced accuracy with the Random Forest before increasing
model size or trying a longer training schedule.
# Multi-task 1D CNN versus Random Forest

## Purpose

Both models describe an **already detected fall** from a two-second IMU window
centered at the labeled impact. They predict two separate properties:

- **Direction:** backward, forward, lateral, or unspecified.
- **Activity context:** fainting, running, sitting, standing, walking, or
  unspecified.

They are not the binary ADL-versus-fall detector. Direction and context are
task-level labels derived from the KFall label-workbook descriptions.

## Multi-task 1D CNN

The deep-learning model receives each trial as a raw `9 x 201` sensor tensor:

- nine channels: AccX/Y/Z, GyrX/Y/Z, and EulerX/Y/Z;
- 201 samples covering -1 to +1 seconds around impact.

It does not receive manually calculated peaks or timing values. Instead, its
three-layer one-dimensional convolutional backbone learns useful temporal
patterns directly from the signals. Global average pooling creates one shared
64-value representation, which feeds two independent softmax classification
heads:

```text
9 sensor channels x 201 samples
              |
     shared 1D convolutional backbone
              |
       64-value shared embedding
          /                 \
 direction head          context head
```

The model is trained for 75 epochs with AdamW (`learning_rate=0.001`, batch
size 64). Its loss is the sum of class-weighted direction and context
cross-entropy losses. Class weights prevent common labels from dominating the
training objective.

## Fair evaluation protocol

Both methods use the same 2,315 fall trials from 32 subjects and five-fold
subject-grouped cross-validation. All trials belonging to a held-out subject
remain out of training. Performance is **balanced accuracy**, which averages
recall across classes and is appropriate for imbalanced labels.

For the CNN, channel mean and standard deviation are fitted only on the
training subjects of each fold. This avoids normalization leakage into the
held-out subjects.

## Results

The values below come from the current saved result files:

- CNN: `results/event_aligned/multitask_1d_cnn/grouped_classification.csv`
- Random Forest: `results/event_aligned/grouped_classification.csv`

| Target | Multi-task 1D CNN | Random Forest | CNN change |
|---|---:|---:|---:|
| Direction | 96.0% +/- 2.4% | 94.9% +/- 2.4% | +1.1 percentage points |
| Context | 79.7% +/- 2.0% | 70.2% +/- 4.9% | +9.5 percentage points |

The CNN is slightly better for direction and substantially better for context.
Its lower context-score variation across held-out folds is also encouraging.
This is evidence that the raw temporal signal contains context information not
fully captured by the Random Forest's summary features; it is not proof of
real-world clinical accuracy.

## Main differences

| Aspect | Multi-task 1D CNN | Random Forest |
|---|---|---|
| Input | Raw nine-channel time series | Hand-engineered event features |
| Learning | Learns temporal patterns automatically | Learns rules over supplied peak, timing, and orientation features |
| Outputs | One shared encoder with direction and context heads | Two independent forests |
| Model configuration | Three Conv1D layers, 64-value embedding | 300 decision trees per target |
| Interpretability | Lower; learned filters are not directly meaningful | Higher; feature importance and tree rules can be inspected |
| Compute | Requires PyTorch and model training | Lightweight CPU training/inference |
| Current result | Best of the two for both targets | Useful interpretable baseline |

## Context-class note

The approved taxonomy has six classes, including `unspecified`. The saved CNN
run contains five context classes (`fainting`, `running`, `sitting`,
`standing`, and `walking`) because none of its retained trials was assigned
`unspecified`. If future data includes stairs, lying, or another unsupported
description, it is mapped to `unspecified` and the CNN context head will
automatically expand to include that class when retrained.

## Result files

- `results/event_aligned/multitask_1d_cnn/fold_metrics.csv`: score for each
  held-out fold.
- `results/event_aligned/multitask_1d_cnn/out_of_fold_predictions.csv`:
  trial-level CNN predictions and confidence values.
- `results/event_aligned/multitask_1d_cnn/run_config.json`: saved CNN settings
  and observed label classes.
- `results/event_aligned/grouped_classification.csv`: Random Forest summary.
