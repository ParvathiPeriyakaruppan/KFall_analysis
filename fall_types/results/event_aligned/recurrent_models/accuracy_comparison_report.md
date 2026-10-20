# ConvLSTM vs 1D CNN+GRU fall-type evaluation

## Evaluation setup

- Data: 2315 impact-aligned fall trials from 32 subjects.
- Input: 9 sensor channels over 201 samples (-1 to +1 second around impact).
- Validation: 5-fold GroupKFold by subject; each subject is held out from training in its fold.
- Preprocessing: per-channel mean and standard deviation fit on training subjects in each fold.
- Targets: task-description-derived direction and context; these are not independently annotated trial labels.
- Training: 75 epochs per fold, batch size 64, seed 42, device `cpu`.
- Metrics: accuracy, balanced accuracy (primary under class imbalance), and macro F1. Reported values are mean ± standard deviation across folds.

## Results

| Model | Target | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| ConvLSTM | direction | 0.962 ± 0.008 | 0.931 ± 0.021 | 0.932 ± 0.019 |
| ConvLSTM | context | 0.810 ± 0.018 | 0.805 ± 0.014 | 0.809 ± 0.023 |
| CNN+GRU | direction | 0.973 ± 0.008 | 0.959 ± 0.015 | 0.954 ± 0.015 |
| CNN+GRU | context | 0.800 ± 0.011 | 0.785 ± 0.022 | 0.792 ± 0.012 |

## Paired fold differences

Difference is CNN+GRU minus ConvLSTM, computed on the same held-out subject folds. Positive values favor CNN+GRU.
| Target | Metric | Mean difference | Fold SD |
|---|---|---:|---:|
| direction | accuracy | +0.010 | 0.004 |
| direction | balanced accuracy | +0.028 | 0.026 |
| direction | macro f1 | +0.022 | 0.015 |
| context | accuracy | -0.010 | 0.018 |
| context | balanced accuracy | -0.020 | 0.017 |
| context | macro f1 | -0.017 | 0.022 |

## Interpretation

Use balanced accuracy as the main comparison because class counts may differ. Fold-to-fold standard deviations describe sensitivity to the held-out subject partition; with five folds they are not a substitute for a larger repeated validation study.
The two heads are evaluated separately. Context and direction labels inherit the limitations of the task-description mapping, so these scores should not be interpreted as clinical or real-world performance.

Artifacts: `fold_metrics.csv` contains per-fold scores; `out_of_fold_predictions.csv` contains held-out trial predictions; `summary_metrics.csv` contains aggregate scores; `fold_*_*.pt` contains fold model weights and training-fold normalization.
