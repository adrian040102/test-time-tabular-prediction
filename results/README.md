# Result evidence

This directory contains the result evidence used in the thesis. The package
includes every usable evaluation that entered the analysis together with the
derived tables needed to inspect the reported comparisons. Intermediate outputs
and execution logs are excluded.

The three main levels of evidence are:

- `evaluations/cell_lift.csv`: all 59,229 usable evaluation cells, including
  fold-level prediction metrics and comparisons with the corresponding
  inductive baseline
- `evaluations/variant_dataset_effects.csv`: 26,401 fold-aggregated effects for
  individual configurations and datasets
- `summaries/variant_model_summary.csv`: the 414 primary
  results for configurations and models reported in the configuration tables

All evaluated configurations and their result values are retained.

## Metric conventions

Prediction error is the quantity minimized by the evaluation: AUC error for
classification and RMSE for regression. Multiclass AUC is the average of the
one-versus-rest class AUCs. Accuracy and log loss are retained as supporting
classification measures. Relative improvement is calculated against the
corresponding inductive baseline. Positive values mean that a configuration
reduced error. Negative values mean that it increased error.

The primary across-dataset result is
`summaries/variant_model_summary.csv`. Its `estimator` column is
`masked_lift_pct_from_means`: fold errors are averaged within each dataset,
relative error reduction is then calculated and relative results are not used
for near-perfect classification baselines. The dataset-level counterpart is
`evaluations/variant_dataset_effects.csv` in `primary_effect_pct`.

The fold-level table also contains unmasked descriptive fields such as
`rel_lift_pct`. Values from different aggregation levels need not be identical.
For example, an all-dataset configuration summary and a comparison restricted
to datasets with both the joint version and train-only version use different
denominators. Column names and sample-size fields identify the relevant unit.

Model identifiers used in the tables are:

| Identifier | Prediction model |
|---|---|
| `bag8` | LightGBM TabArena bagging ensemble |
| `tabicl` | TabICLv2 |
| `tabpfn` | TabPFNv2 |

## Directory guide

| Directory | Contents |
|---|---|
| `evaluations/` | Complete usable evaluation cells and final dataset-level effects. |
| `summaries/` | Primary configuration summaries, normalized rankings and explicitly labelled secondary family summaries. |
| `metadata/` | Coverage counts, dataset characteristics and split/seed disclosure. |
| `joint_train_only/` | Dataset-level and model-level results for ten joint versions and their train-only versions. |
| `shift/` | Cross-dataset domain-AUC trends, controlled synthetic shift and within-source TabReD window comparisons. |
| `model_comparison/` | The final common-fold comparison: 125 configurations, 17 families and 31 data sources available for all three models. |
| `reliability/` | Estimator, source-weighting, influence, near-perfect-baseline, leave-one-out and fold-consistency checks. |
| `case_studies/` | Dataset-level and family-level tables used for the reported case studies and Discussion diagnostics. |
| `exploratory/` | The primary dataset-trait correlation table, trait dictionary, analysis specification and verification summary. |
| `verification/` | Machine-readable verification status for the primary Chapter 4 evidence. |

`summaries/reliable_positive_configurations.csv` intentionally has a header but no
data rows. No configuration met every reliability criterion.

`case_studies/discussion_case_diagnostics.json` contains explanatory diagnostics
used in the dataset discussion. Test labels were used only after pseudo-label
generation to measure the accuracy of the selected pseudo-labels. They were not
available to the evaluated method.

`SHA256SUMS.csv` lists the byte size, row count and SHA-256 digest of every
other evidence file in this directory. It can be used to check that a downloaded
copy has not changed.

The original real-world datasets are not redistributed. See the repository's
`DATASETS.md` for acquisition information and the repository README for the
experiment protocol and environment limitations.
