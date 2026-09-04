# KFall Event Categorization and Plot Conclusions

This document explains how the fall events were categorized and what can be
reasonably inferred from the generated plots. The analysis uses the KFall label
workbooks, `fall_type_metadata.csv`, and the plots in `plots/`.

## 1. What was categorized

The sensor CSV files contain time-series measurements, but they do not contain
columns named `direction` or `context`. These categories come from the
task-level `Description` field in each subject's label workbook. The labels are
therefore experimental situation metadata, not values that change at every
sample.

There are 2,319 fall trials from task IDs 20-34, corresponding to F01-F15.
The description is retained for every trial in `fall_type_metadata.csv`, so the
inference can be audited.

## 2. Direction categorization

The description was converted to a direction using these keyword rules:

| Description contains | Direction assigned |
|---|---|
| `backward`, `backwards`, or `back fall` | backward |
| `lateral`, `sideways`, `sideway`, or `side fall` | lateral |
| `forward` or `front fall` | forward |
| none of the above | unspecified |

The checks are applied in the order backward, lateral, forward. The source
description is still kept; the derived value is only a grouping convenience.

Final direction counts:

| Direction | Trials |
|---|---:|
| forward | 1,077 |
| lateral | 626 |
| backward | 465 |
| unspecified | 151 |

`unspecified` is preferable to guessing. For example, a description such as
“fall while walking ... caused by fainting” does not state whether the fall was
forward, backward, or lateral.

## 3. Context categorization

The description was converted to a situation/context using these rules:

| Description contains | Context assigned |
|---|---|
| `faint` or `syncope` | fainting |
| `stair` or `stairs` | stairs |
| `run`, `running`, or `jog` | running |
| `walk` or `walking` | walking |
| `sit`, `sitting`, or `chair` | sitting |
| `stand`, `standing`, or `get up` | standing |
| `lie`, `lying`, or `bed` | lying |
| none of the above | unspecified |

The rules are applied in the order shown. Thus a description mentioning both
walking and fainting is grouped as `fainting`, because the cause/situation of
the event is more specific than the movement state.

Final context counts:

| Context | Trials |
|---|---:|
| fainting | 766 |
| walking | 623 |
| sitting | 470 |
| standing | 312 |
| running | 148 |

Some task IDs have slightly different descriptions across subjects. The
implementation does not overwrite these differences with a forced task-wide
label. This is why the metadata should be inspected before making a strict
task-level claim.

## 4. Signals plotted

Each PNG contains nine panels:

- Acceleration: `AccX`, `AccY`, `AccZ`, in g.
- Angular velocity: `GyrX`, `GyrY`, `GyrZ`, in degrees/second.
- Orientation: `EulerX`/roll, `EulerY`/pitch, and `EulerZ`/yaw, in degrees.

The horizontal axis is elapsed time from the beginning of each recording. Each
group plot contains thin traces from a deterministic subset of individual
trials and a bold mean calculated from all trials. Trials are interpolated onto
the shortest recording duration within that group. The individual overlay is
limited for readability and rendering speed; it does not affect the mean.

## 5. Conclusions from the plots

### Acceleration

- The signed acceleration axes do not show one universal direction-specific
  waveform in the raw elapsed-time plots.
- Their means are often close to zero or near a gravity-dependent offset,
  while individual traces contain large late-recording excursions.
- This is expected because the sensor orientation differs between subjects and
  trials. A physical forward fall can project onto different sensor axes.
- Therefore, raw `AccX`, `AccY`, or `AccZ` alone should not be interpreted as a
  universal forward/backward/lateral indicator.

### Gyroscope

- Individual gyroscope traces show strong trial-to-trial angular motion,
  especially in `GyrX` and `GyrY` for the forward and fainting groups.
- The signed group means are much smaller than the individual peaks because
  positive and negative rotations cancel each other.
- This suggests that angular-velocity magnitude, peak magnitude, or
  event-aligned signed features may be more useful than a raw signed-axis mean.

### Orientation angles

- `EulerX`/roll shows a visible average change in the forward and fainting
  examples, but the spread is wide.
- `EulerY`/pitch and `EulerZ`/yaw have weaker and less consistent group-level
  changes in the displayed plots.
- Orientation angles appear useful as supporting context, but the plots do not
  justify using one Euler axis as a standalone direction label.

### Context differences

- Fainting plots contain long quiet regions followed by sparse, high-amplitude
  excursions. This is consistent with a slower or less synchronized event
  population than walking/running falls.
- Forward falls show stronger average orientation change than a simple signed
  acceleration pattern. This supports examining direction through orientation,
  acceleration magnitude, and angular-velocity magnitude together.
- Different contexts visibly change timing and variability. A single global
  threshold or average across all contexts can hide these differences.

## 6. Important limitations

These plots are descriptive, not proof of a direction or context classifier.

1. The recordings are aligned at time zero, not at labeled fall onset or
   impact. A fall occurring at different times in different recordings will be
   blurred in a group mean.
2. Each group is resampled to its shortest recording duration, which makes
   comparisons convenient but removes absolute-duration differences.
3. Signed sensor axes depend on wearable placement and body orientation. Axis
   sign is not automatically equivalent to physical forward/backward/lateral
   motion.
4. The context labels describe the experimental task. They are not proof that
   every individual motion inside the recording has the same context.
5. The `unspecified` direction group must be reviewed from its original
   descriptions before it is used for supervised classification.

## 7. Event-aligned analysis results

The second analysis has now been completed using the onset and impact frames
from `label_data`. Every valid trial was aligned to impact, with impact at 0 s,
over a window from -1 s to +1 s. The outputs are in `event_aligned/`:

- `event_aligned_features.csv` — 2,315 trial-level feature rows.
- `separability_tests.csv` — Kruskal-Wallis tests and epsilon-squared effect
  sizes for every feature by direction and context.
- `grouped_classification.csv` — subject-grouped random-forest balanced
  accuracy, using GroupKFold so trials from one subject do not leak between
  training and testing.
- `direction/*.png` and `context/*.png` — mean +/- standard deviation plots for
  all raw axes plus acceleration magnitude, gyroscope magnitude, and
  orientation change.

For each direction/context group, the feature table includes:

- acceleration magnitude and peak timing,
- gyroscope magnitude and peak timing,
- orientation change from pre-event baseline,
- onset-to-impact duration,
- peak and median absolute values for every raw axis.

## 8. Separability findings

| Target category | Trials | Subjects | Grouped balanced accuracy | Interpretation |
|---|---:|---:|---:|---|
| Direction | 2,315 | 32 | 94.9% +/- 2.4% | Strong evidence that the extracted event features contain directional information. |
| Context | 2,315 | 32 | 70.2% +/- 4.9% | Moderate separation; context is substantially harder and overlaps between groups. |

The direction result is supported by very small Kruskal-Wallis p-values across
many features. The largest directional effect sizes in the current table come
from orientation-change features and gyroscope-magnitude features, while
acceleration-magnitude effects are smaller. This indicates that direction is
more strongly expressed through rotational/orientation behavior than through a
single signed acceleration axis.

Context is not cleanly separable. Fainting, sitting, walking, running, and
standing share overlapping sensor behavior, and the context label describes the
experimental scenario rather than a unique physical waveform. The 70.2%
balanced accuracy is evidence of useful context information, not a claim that
context can be identified reliably in every trial.

These results are stronger than visual inspection alone because the classifier
was evaluated with subject-grouped folds. They should still be treated as
exploratory: feature selection, calibration, and an independent held-out test
set would be needed before claiming a deployable direction/context classifier.