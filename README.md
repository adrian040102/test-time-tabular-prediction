# Leveraging Test-Time Information for Improved Tabular Prediction

This repository contains the implementation and final analysis evidence for
the master's thesis *Leveraging Test-Time Information for Improved Tabular
Prediction*.

The study examines whether unlabeled test features can improve tabular
predictions under dataset shift. Methods may use the training features, training
labels and test features. They never receive the test labels.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/` | Data loading, leakage protection, methods, prediction-model wrappers, pipeline code and evaluation utilities. |
| `scripts/run_experiment.py` | Entry point for one experiment. |
| `scripts/run_sweep.py` | Simple local sweep runner with resumable CSV output. |
| `scripts/slurm/` | Fold-aware manifest generation, task execution and result collection for large sweeps. |
| `scripts/preprocessing/` | Dataset-specific TabReD preprocessing. |
| `configs/experiments/` | The three seed-0 configurations for the reported model sweeps. |
| `data/` | Metadata for the 92 evaluated dataset versions and stored data for five controlled synthetic versions. |
| `results/` | Complete usable evaluation evidence, derived analysis tables and verification records. |
| `tests/` | Leakage, pipeline-contract, model, method and synthetic mechanism checks. |

The study covers 17 method families, 145 study configurations, 82 original
datasets, 10 additional versions of four TabReD datasets and three prediction
models. The three experiment configurations contain the compatible subset for
each model.

## Environment

Python 3.10 through 3.12 is supported. Python 3.10 is the conservative choice for
reconstructing the reported environment. Python 3.13 is not supported by
AutoGluon 1.4.0. The result-relevant versions reported in the thesis are:

- `lightgbm` 4.6.0, used through `autogluon.tabular` 1.4.0
- `tabpfn` 2.2.1
- `tabicl` 2.1.1
- `scikit-learn` 1.7.2
- `openTSNE` 1.0.4
- `scipy` 1.15.3

`requirements.lock` records the direct versions and installs the common
environment after TabPFN is installed as described below. It is a curated list,
not a complete lock of every transitive dependency. In particular, the
appropriate PyTorch build depends on the target CPU or CUDA environment.

TabPFN 2.2.1 declares `scikit-learn<1.7`, while the recorded and
manuscript-reported environment uses scikit-learn 1.7.2. The study ran that
combination successfully, but a single standard dependency-resolution command
cannot create it.

From the repository root, create and activate a virtual environment, install a
PyTorch build suitable for the machine and then run the following sequence.
The `requirements.lock` command restores the recorded scikit-learn 1.7.2 version
after pip has installed TabPFN and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install "tabpfn==2.2.1"
python -m pip install -r requirements.lock
```

`pip check` will report the known TabPFN/scikit-learn metadata mismatch in this
reconstructed environment. This is expected and is disclosed here rather than
replacing the manuscript-consistent scikit-learn version.

Dataset acquisition and testing also require `openml`, `folktables` and
`pytest`, which are not part of the curated lock file. Install them when needed:

```bash
python -m pip install "openml>=0.14" "folktables>=0.0.12" "pytest>=7"
```

Run the code directly from the repository checkout. Follow the installation
sequence above because `pip install .` alone does not reconstruct the recorded
environment.

## Quick local check

The following command uses one of the included synthetic datasets and writes one
result row under `results/quick_check/`:

```bash
python scripts/run_experiment.py --dataset synthetic_shift_1.0 --model lightgbm --method inductive_baseline --seed 0 --results-dir results/quick_check
```

The expected output file is:

```text
results/quick_check/synthetic_shift_1.0_inductive_baseline_lightgbm_s0.csv
```

This is a functional check of the local pipeline. It does not reproduce the
reported eight-model LightGBM ensemble.

A small comparison can be run with:

```bash
python scripts/run_sweep.py --datasets synthetic_shift_0.0 synthetic_shift_1.0 --models lightgbm --methods inductive_baseline joint_freq_combined --seeds 0 --results-dir results/quick_check --tag quick_check
```

## Reported experiment protocol

The reported evaluations use:

- an eight-fold LightGBM ensemble implemented through AutoGluon
- TabPFNv2 with eight ensemble members
- TabICLv2 with two ensemble members

All main benchmark and gap-widening evaluations use seed 0. The 51 TabArena
datasets use their first three predefined folds. The other original datasets
use one fixed split. The additional TabReD versions keep the latest test period
fixed and vary the earlier training period. The separate designed mechanism
checks use five seeds.

The main configurations are:

- `configs/experiments/tabarena_final_lgb_bag8.yaml`
- `configs/experiments/tabarena_final_tabicl.yaml`
- `configs/experiments/tabarena_final_tabpfn.yaml`

These three seed-0 configurations also include the applicable gap-widened
TabReD versions. The fold-aware large-sweep path is:

1. `scripts/slurm/generate_manifest.py`
2. `scripts/slurm/worker.py`
3. `scripts/slurm/collect_results.py`

For example, a one-cell LightGBM manifest using an included synthetic dataset
can be generated and executed locally with:

```bash
python scripts/slurm/generate_manifest.py --config configs/experiments/tabarena_final_lgb_bag8.yaml --datasets synthetic_shift_1.0 --methods inductive_baseline --batch-size 1 --output scripts/slurm/manifest_quick_check.csv
python scripts/slurm/worker.py --manifest scripts/slurm/manifest_quick_check.csv --results-dir results/runs/quick_check
python -X utf8 scripts/slurm/collect_results.py --runs-dir results/runs/quick_check --manifest scripts/slurm/manifest_quick_check.csv --output results/quick_check_bag8.csv
```

`python -X utf8` prevents an encoding error on Windows consoles that cannot
print the collector's final status symbol.

With no scheduler environment variable, `worker.py` executes task 0. Larger
manifests assign task IDs through `SLURM_ARRAY_TASK_ID` and can be run locally or
as a scheduler array. Omitting the dataset and method filters generates the
complete model manifest. `run_sweep.py` is useful for smaller local checks, but
it does not expand the complete fold-aware protocol encoded in the reported
configurations.

For large runs, point `OPENML_CACHE_DIRECTORY`, `THESIS_DATA_CACHE`,
`FOLKTABLES_CACHE_DIR` and `XDG_CACHE_HOME` to writable locations with enough
space. Auxiliary random-pipeline draw specifications are intentionally absent
because that composition analysis is outside the reported protocol. The worker
therefore registers no random-pipeline methods.

## Data availability

The repository includes the train and test parquets for the five
author-generated controlled-shift versions. It does not redistribute any
real-world dataset. Metadata are included so that the final dataset roster,
feature types, split sizes and shift descriptions remain inspectable.

See [DATASETS.md](DATASETS.md) for the data groups, acquisition entry points
and redistribution boundary. External datasets remain subject to their source
providers' licenses and terms.

## Result availability

`results/` contains all 59,229 usable evaluation cells that entered the final
analysis, the 26,401 effects for individual configurations and datasets and the
414 primary summaries for configurations and models. It also contains the
derived evidence used
for the comparisons between joint and train-only versions, shift analysis,
model comparisons, reliability analysis, case studies and exploratory analysis.
See [results/README.md](results/README.md) for the file guide and metric
conventions. [results/SHA256SUMS.csv](results/SHA256SUMS.csv) records the size,
row count and SHA-256 digest of every evidence file.

Only files that entered the reported analysis are included. Intermediate outputs
and execution logs are excluded. The public
package supports direct inspection of the reported evidence and rerunning
experiment cells, but it does not provide a one-command rebuild of every
manuscript table. Recreating the complete benchmark also requires obtaining the
external datasets and substantial CPU and GPU compute.

`results/summaries/reliable_positive_configurations.csv` intentionally contains a
header and no data rows because no evaluated configuration met every reliability
criterion.

## Tests

The following data-independent core command passes for this repository snapshot:

```bash
python -m pytest -p no:cacheprovider -q tests/test_leakage.py tests/test_pipeline_contracts.py tests/test_joint_train_only_parity.py tests/test_preserve_originals.py
```

With the documented dependencies installed, the core command completed with
504 passed tests and 6 skips. When AutoGluon is unavailable, one eight-fold
LightGBM integration test is skipped, so the numbers of passed and skipped tests
may differ across environments.

The complete suite also includes integration tests that require the excluded
real-world Parquet files, so it is not expected to pass on a metadata-only
checkout. Use the core command above unless you have prepared the external
datasets described in `DATASETS.md`.

## Reproducibility notes

- `requirements.lock` is not a complete platform-level environment freeze.
- TabPFNv2 and TabICLv2 may download model weights on first use.
- Joint t-SNE can vary across processor types even when the data, settings and
  random seed are unchanged.
- Dataset providers may revise hosted files or access procedures.
- Full sweeps support parallel execution and require substantial compute.

## Citation

If you use this repository, please cite:

> Adrian Scheibelhut. *Leveraging Test-Time Information for Improved Tabular
> Prediction*. Master's thesis, University of Mannheim, 2026.

## License

Unless stated otherwise, the author's original source code, documentation,
result tables and controlled synthetic fixtures are available under the
[MIT License](LICENSE). Modified or reimplemented third-party components are
identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and retain their
applicable licenses. Third-party datasets, model weights and dependencies
remain subject to their respective terms and licenses.
