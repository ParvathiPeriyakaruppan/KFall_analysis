# KFall Fall-Type Analysis

This folder contains the fall-direction and fall-context analysis for the KFall
dataset. It is intentionally self-contained: scripts, documentation, and
generated outputs are organized below this folder only.

## Folder layout

```text
fall_types/
├── scripts/
│   ├── plot_fall_types.py
│   └── event_aligned_analysis.py
├── docs/
│   ├── parameter_analysis.md
│   ├── event_categorization_and_plot_conclusions.md
│   └── binary_detection_and_fall_type_plan.md
├── agents/
│   └── fall_type_visualization.agent.md
└── results/
		├── metadata/fall_type_metadata.csv
		├── plots/                 # elapsed-time plots
		└── event_aligned/         # event plots, features, and tests
```

## Requirements

Use the project virtual environment and install the dependencies from the
project requirements file. The dataset must contain `sensor_data/` and
`label_data/`. The scripts automatically use the downloaded KaggleHub cache
when present. To use another location, set `KFALL_DATASET_ROOT` to the parent
directory containing both folders.

## Run the analyses

From `KFall_analysis`:

```text
python fall_types/scripts/plot_fall_types.py
python fall_types/scripts/event_aligned_analysis.py
```

The second command can be accelerated for plot-only reruns with:

```text
$env:KFALL_SKIP_TESTS = "1"
$env:KFALL_COMPARISON_ONLY = "1"
python fall_types/scripts/event_aligned_analysis.py
```

## Outputs

`results/plots/` contains 15 task plots and direction/context plots. Each plot
has nine panels for acceleration X/Y/Z, gyroscope X/Y/Z, and Euler X/Y/Z.

`results/event_aligned/` contains plots aligned to impact, where impact is
`0 seconds` over a `-1` to `+1 second` window. It includes:

- `direction/` and `context/`: separate event-aligned group plots.
- `comparison/`: six four-panel comparison images. Each image overlays all
	direction or context groups in different colors for acceleration, gyroscope,
	or orientation.
- `event_aligned_features.csv`: 2,315 trial-level feature rows.
- `separability_tests.csv`: Kruskal-Wallis tests and effect sizes.
- `grouped_classification.csv`: subject-grouped random-forest balanced accuracy.

## Interpretation

Direction and context are task-level categories derived from the label
workbook `Description` field; they are not per-sample sensor labels. The current
event-aligned features give approximately 94.9% subject-grouped balanced
accuracy for direction and 70.2% for context. Direction is promising, while
context should be treated as a lower-confidence supplementary output.

The current binary detector still answers only ADL versus fall. The planned
extension is documented in `docs/binary_detection_and_fall_type_plan.md`: reuse
the existing model backbone and add lightweight direction and context heads
after the binary fall decision.