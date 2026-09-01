# Data availability

The reported study uses 92 dataset versions: 82 original datasets and 10
additional versions derived from four TabReD datasets. This repository includes
metadata for all 92 versions. It includes data files only for five
author-generated controlled-shift versions.

No real-world records are redistributed through this repository. Git continues
to ignore downloaded and processed data. Each external dataset remains governed
by the license, access rules and terms published by its provider.

## Dataset groups

| Group | Versions | Data included | Acquisition or construction path |
| --- | ---: | --- | --- |
| TabArena | 51 | Metadata only | OpenML Study 457 through `scripts/download_tabarena.py`, followed by `scripts/augment_tabarena_folds.py`. |
| Folktables/ACS | 7 | Metadata only | `src/data/folktables.py` and the cleaning paths in `scripts/build_clean_datasets.py`. |
| Standalone shift datasets | 6 | Metadata only | Loaders for Bike Sharing, BRFSS Diabetes, Diabetes 130-US Hospitals, Gas Sensor Drift, Lending Club and Yandex SHIFTS Weather under `src/data/`. |
| Tschalzev et al. datasets | 3 | Metadata only | OpenML identifiers and loading logic in `src/data/openml_datasets.py`. |
| Synthetic datasets | 9 | Five controlled-shift train/test pairs and metadata for four other generated designs | Generator in `src/data/synthetic.py`. |
| TabReD original datasets | 6 | Metadata only | Source-specific scripts under `scripts/preprocessing/`. |
| Additional TabReD versions | 10 | Metadata only | `scripts/characterize_gap_widening.py` and `scripts/build_gap_widened_tabred.py`. |

The dataset metadata record the final feature partition, task type, split sizes
and shift description. Metadata do not grant redistribution rights and should
not be treated as a substitute for the provider's documentation.

## Included synthetic fixtures

The following directories contain both `train.parquet` and `test.parquet`:

- `data/datasets/synthetic_shift_0.0/`
- `data/datasets/synthetic_shift_0.5/`
- `data/datasets/synthetic_shift_1.0/`
- `data/datasets/synthetic_shift_3.0/`
- `data/datasets/synthetic_shift_5.0/`

Each version has 10,000 training rows and 10,000 test rows. The training data and
prediction rule are fixed. Only the test-feature shift strength changes. The
stored split was generated with seed 42, while the reported model evaluation
uses seed 0.

## TabArena

The 51 TabArena datasets come from OpenML Study 457. To download the dataset
payloads and add the official fold layer, run from the repository root:

```bash
python scripts/download_tabarena.py --overwrite
python scripts/augment_tabarena_folds.py --overwrite
```

The first script writes a local parquet representation. The second writes the
full OpenML-order data and fold indices required by `fold_idx`. The reported
study uses the first three predefined folds, as encoded by the final sweep
configuration. `--overwrite` is needed when the metadata-only dataset directories
already exist.

`data/tabarena/datasets/_index.json` records the preparation run from which the
metadata were copied. Its successful-download entries describe that preparation
snapshot. They do not mean that the corresponding parquet payloads are bundled
in this repository.

## Other real-world datasets

Some loaders can download source data directly. Others expect source exports
that the user must obtain separately. The relevant entry points are:

- `scripts/build_clean_datasets.py` for the supported cleaned datasets
- `src/data/folktables.py` for Folktables and ACS data
- the individual loaders in `src/data/` for UCI, CDC BRFSS, Lending Club,
  Yandex SHIFTS and OpenML sources
- `scripts/preprocessing/tabred_<name>.py` for TabReD datasets.

When a metadata-only output directory already exists, use the relevant
`--overwrite` option. API-backed construction may also require
`--allow-refetch`.

TabReD preprocessing expects separately obtained source files under
`data/tabred/raw/<dataset>/` by default. See
`thesis/docs/tabred_kaggle_setup.md` and the individual script headers for source
identifiers and expected filenames. Use `--help` to view the available options
and supply a different input directory.

The preparation steps can involve large downloads, source-specific access
requirements and substantial disk space. Before downloading or processing any
external dataset, consult its current source page and comply with its terms.

## Git boundary

The repository's `.gitignore` excludes all downloaded and processed real-world
data. Only metadata and the five controlled synthetic fixtures are explicitly
allowed. Use `git ls-files` to confirm that no additional data file is tracked.
