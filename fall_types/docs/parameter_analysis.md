# Parameter Analysis Before Raw Data Download

This analysis uses the existing processed file
`../fall_pattern_analysis/paper_threshold_validation/pattern_results.csv`.
It does not replace the raw signal analysis: acceleration, gyroscope, and
Euler-angle distributions still require the extracted sensor CSV files.

## What the dataset can label

The fall trials are task IDs 20-34, corresponding to F01-F15. Direction and
context are task-level metadata. They are not separate labels in the processed
results file and should not be treated as changing within a trial.

The working task map is:

| Task IDs | Direction groups | Context groups |
|---|---|---|
| 20-22 | forward, backward, lateral | sitting |
| 23-24 | forward, backward | standing |
| 25-27 | forward, backward, lateral | fainting/sitting |
| 28-30 | forward, backward, lateral | walking |
| 31-33 | forward, backward, lateral | running |
| 34 | unspecified until the workbook description is checked | stairs |

Task descriptions in the label workbooks should be treated as authoritative.
The plot script preserves them in `fall_type_metadata.csv` and only uses the
map above as a fallback.

## Existing detector results

There are 2,319 fall trials and 2,717 ADL trials in the processed file.

| Parameter/rule | Fall result | ADL result | Interpretation |
|---|---:|---:|---|
| Rule A: `ACC_M < 0.8 g` | 100.0% detected | 82.6% triggered | Very sensitive, poor discrimination |
| Rule B: Rule A + `VV > 0.3 m/s` + angle confirmation | 95.6% detected | 65.1% triggered | Better, but still many ADL false positives |
| Rule C: gyro rule on F06-F08 | 100.0% detected, n=461 | Not evaluated | Promising subset result, no specificity conclusion |

Rule B detects 101 fewer fall trials than Rule A. Its mean lead time is about
648 ms versus 936 ms for Rule A, but the lead-time distribution is broad, so
the mean should not be interpreted as a fixed warning time.

## Direction and context observations

Rule A is 100% for every fall task, so it cannot distinguish direction or
context. Rule B varies substantially:

| Direction/context | Trials | Rule B detection | Mean lead time |
|---|---:|---:|---:|
| Forward/fainting | 154 | 62.3% | 628 ms |
| Forward/sitting | 155 | 77.4% | 624 ms |
| Forward/standing | 155 | 98.1% | 609 ms |
| Forward/walking | 154 | 99.4% | 636 ms |
| Forward/running | 148 | 100.0% | 1,850 ms |
| Backward/fainting | 154 | 100.0% | 357 ms |
| Lateral/fainting | 153 | 100.0% | 344 ms |

The clearest current finding is a **forward, slow/sitting or fainting weakness**
in the acceleration-plus-vertical-velocity-plus-angle rule. This supports
separating direction from context in later plots: direction-only averages may
hide the fact that the same direction behaves differently when sitting,
walking, running, or fainting.

## Parameters to inspect when raw data arrives

For each task, direction group, and context group, compare:

- Acc X/Y/Z: baseline, minimum, impact peak, peak timing, and axis dominance.
- Gyr X/Y/Z: peak angular velocity, peak timing, and whether one axis dominates.
- Euler X/Y/Z: orientation change and pre-impact angle spread.
- `ACC_M = sqrt(AccX^2 + AccY^2 + AccZ^2)` and its relationship to the raw axes.
- Vertical velocity and its relationship to acceleration and orientation.
- Onset-to-impact duration and lead time, reported with median and interquartile range.

These measurements require raw timestamps and labeled onset/impact frames and
are therefore intentionally deferred until `sensor_data/` and `label_data/`
finish downloading.