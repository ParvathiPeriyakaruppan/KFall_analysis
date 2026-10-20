# Deep-learning overfitting audit

Compared saved five-fold subject-held-out models on 2315 trials from 32 subjects.
Training and held-out scores use the same fold model and training-fold normalization. A large train-to-held-out gap can indicate overfitting.
Balanced accuracy is the main metric because class frequencies differ. Values are mean ± standard deviation across folds.

| Model | Target | Split | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| 1D CNN | direction | train | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.998 ± 0.002 |
| 1D CNN | direction | held-out | 0.974 ± 0.008 | 0.960 ± 0.024 | 0.959 ± 0.016 |
| 1D CNN | context | train | 0.953 ± 0.013 | 0.960 ± 0.012 | 0.947 ± 0.018 |
| 1D CNN | context | held-out | 0.797 ± 0.031 | 0.797 ± 0.020 | 0.778 ± 0.034 |
| ConvLSTM | direction | train | 0.994 ± 0.004 | 0.990 ± 0.008 | 0.989 ± 0.007 |
| ConvLSTM | direction | held-out | 0.962 ± 0.008 | 0.931 ± 0.021 | 0.932 ± 0.019 |
| ConvLSTM | context | train | 0.947 ± 0.014 | 0.952 ± 0.015 | 0.949 ± 0.016 |
| ConvLSTM | context | held-out | 0.810 ± 0.018 | 0.805 ± 0.014 | 0.809 ± 0.023 |
| CNN+GRU | direction | train | 0.997 ± 0.002 | 0.997 ± 0.002 | 0.997 ± 0.003 |
| CNN+GRU | direction | held-out | 0.973 ± 0.008 | 0.959 ± 0.015 | 0.954 ± 0.015 |
| CNN+GRU | context | train | 0.974 ± 0.017 | 0.979 ± 0.012 | 0.978 ± 0.013 |
| CNN+GRU | context | held-out | 0.800 ± 0.011 | 0.785 ± 0.022 | 0.792 ± 0.012 |

## Train-to-held-out balanced-accuracy gap

Gap = training-fold mean minus held-out-fold mean; larger positive values indicate a wider generalization gap.
| Model | Target | Gap |
|---|---|---:|
| 1D CNN | direction | +0.039 (+3.9 percentage points) |
| 1D CNN | context | +0.162 (+16.2 percentage points) |
| ConvLSTM | direction | +0.059 (+5.9 percentage points) |
| ConvLSTM | context | +0.147 (+14.7 percentage points) |
| CNN+GRU | direction | +0.039 (+3.9 percentage points) |
| CNN+GRU | context | +0.194 (+19.4 percentage points) |

The gap is a diagnostic rather than a formal significance test. Labels are derived from task descriptions, so the audit measures generalization to unseen participants under this dataset's labeling scheme.
