# Random Forest Classification Explained

## Aim of the model

The aim is to determine whether sensor measurements contain enough information
to describe an **already detected fall**. The model answers two questions:

1. **Which direction was the fall?** Forward, backward, lateral, or
   unspecified.
2. **What was the activity context?** Fainting, running, sitting, standing,
   walking, or unspecified.

This model is not the binary fall detector. It does not decide whether an event
is a fall. It provides additional information after a fall has been identified.
Direction and context are trained as two separate classification tasks, using
two separate Random Forest models.

## What is given to the model

Each labeled fall trial is aligned around impact:

- Impact is defined as `0 seconds`.
- The time window is `-1` to `+1 seconds`.
- Every trial is interpolated onto the same fixed time grid.

The script converts each trial into one row of numerical features from the
accelerometer, gyroscope, and orientation signals. The features include:

- Fall-onset-to-impact duration.
- Acceleration and gyroscope magnitude peaks and peak timing.
- Median absolute values of derived signals and raw sensor axes.
- Orientation change from the pre-event baseline.

These rows are stored in
`results/event_aligned/event_aligned_features.csv`. The input features are the
sensor-derived values. `direction` and `context` are the target labels, while
identifiers such as subject and trial ID are excluded from the input.

## What is used for training

For each target, the Random Forest learns from:

```text
Sensor features + known labels from training subjects
```

There are two training targets:

```text
Sensor features -> direction
Sensor features -> context
```

Each model contains 300 decision trees. The trees learn different feature-based
rules, and the forest combines their votes to select the predicted class. The
class weighting is balanced so that frequent classes do not dominate the
learning process.

The models use `random_state=42` for reproducibility and `n_jobs=-1` to use all
available CPU cores.

## What is used for testing

Testing uses sensor features and labels from subjects that were not used to
train that fold. The split uses five-fold `GroupKFold` with `subject` as the
group:

- Four subject groups are used for training.
- One subject group is held out for testing.
- The process repeats five times so each group is tested once.

All trials from one subject stay in the same fold. Therefore, the model cannot
memorize a person's particular movement patterns and is tested on unseen
subjects. The held-out predictions from all five rounds are used to calculate
the reported result.

## How performance is measured

The model is evaluated with **balanced accuracy**, which averages the recall of
each class. This is important when some directions or contexts have more trials
than others.

The reported scores in
`results/event_aligned/grouped_classification.csv` come from held-out testing
predictions, not from the training data. The current results are approximately:

- `94.9%` balanced accuracy for direction.
- `70.2%` balanced accuracy for context.

The direction result is stronger. Context classification is useful as a
supplementary output but should be treated with more caution.

## What happens after testing

After cross-validation has measured performance, a final model may be trained
using all labeled trials. This model is intended for future predictions on new
fall events. It must not be used to report the validation score, because a
model trained on all subjects has no independent test subjects left.

## What the model learns

The Random Forest learns associations between sensor patterns and the labels,
including:

- The size of acceleration and gyroscope peaks.
- The timing of peaks relative to impact.
- The amount of body-orientation change.
- Combinations of values across sensor axes.

It does not understand the meaning of words such as "forward" or "walking".
It learns from numerical features paired with those labels in the dataset.

## Important limitation

Direction and context are derived from task descriptions in the label workbook.
They are task-level categories rather than independently measured per-sample
sensor labels. The results should therefore be interpreted as how well the
sensor features reproduce these dataset categories.

## Main files

- `scripts/event_aligned_analysis.py`: feature extraction, training, and
  subject-grouped testing.
- `results/event_aligned/event_aligned_features.csv`: model input features.
- `results/event_aligned/grouped_classification.csv`: held-out test scores.
- `results/event_aligned/random_forest/`: saved fold models, predictions, and
  feature lists when the reproducible model artifacts are generated.
