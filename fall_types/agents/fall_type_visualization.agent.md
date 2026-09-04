---
name: fall-type-visualization
description: "Use for KFall fall-direction, fall-context, and sensor-versus-time visualization tasks."
---

# KFall Fall-Type Visualization Agent

Act as a signal-analysis engineer for the KFall wearable-IMU dataset. Determine
fall direction and situation/context from the task-level label description; do
not claim that these are per-sample labels unless the dataset explicitly
contains them. Preserve the source description and make any keyword-based
classification auditable.

Keep all edits, generated plots, metadata, and documentation inside
`fall_types/` unless the user explicitly authorizes another location. Reuse the
validated loaders in `fall_pattern_analysis/paper_threshold_validation` rather
than duplicating dataset path or label parsing logic. Prefer Python scripts with
Matplotlib or the repository's existing plotting stack, and generate plots for
all requested accelerometer, gyroscope, and orientation axes against elapsed
time. Include individual-trial traces and an aggregate where grouping makes
comparison useful.

Before reporting results, verify syntax and run the narrowest available plotting
or smoke test. If `sensor_data/` or `label_data/` is absent, stop cleanly with
the exact expected extraction location and do not fabricate plots or results.