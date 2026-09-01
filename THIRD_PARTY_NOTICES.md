# Third-party notices

This repository contains modified or reimplemented portions based on the
Apache-2.0-licensed projects identified below. The full Apache License 2.0 is
included at `LICENSES/Apache-2.0.txt`. These notices do not change the MIT terms
for the repository's original material.

## TabReD preprocessing

The following files adapt preprocessing choices from the Yandex Research
TabReD project:

- `scripts/preprocessing/_common.py`
- `scripts/preprocessing/tabred_cooking_time.py`
- `scripts/preprocessing/tabred_delivery_eta.py`
- `scripts/preprocessing/tabred_ecom_offers.py`
- `scripts/preprocessing/tabred_homecredit.py`
- `scripts/preprocessing/tabred_homesite.py`
- `scripts/preprocessing/tabred_maps_routing.py`
- `scripts/preprocessing/tabred_sberbank.py`
- `scripts/preprocessing/tabred_weather.py`

Upstream project and preprocessing sources:

- <https://github.com/yandex-research/tabred>
- <https://github.com/yandex-research/tabred/tree/main/preprocessing>

Upstream license: Apache License 2.0.

Modifications: the preprocessing logic was translated and integrated into this
repository's pandas/parquet dataset format, command-line interfaces, metadata
schema, temporal split handling and sampling workflow. The individual file
headers identify the corresponding upstream scripts and the preserved choices.

## AutoGluon LightGBM behavior

`src/models/lgb_tabarena.py` reimplements the adaptive early-stopping formula
and follows LightGBM configuration choices from AutoGluon v1.4.0. It adapts
them to this repository's model-wrapper and evaluation contracts.

Upstream sources:

- <https://github.com/autogluon/autogluon/tree/v1.4.0>
- <https://github.com/autogluon/autogluon/blob/v1.4.0/core/src/autogluon/core/models/_utils.py>
- <https://github.com/autogluon/autogluon/blob/v1.4.0/tabular/src/autogluon/tabular/models/lgb/lgb_model.py>
- <https://github.com/autogluon/autogluon/blob/v1.4.0/tabular/src/autogluon/tabular/models/lgb/hyperparameters/parameters.py>

Upstream license: Apache License 2.0.

The AutoGluon v1.4.0 `NOTICE` file states:

> AutoML for Text, Image, and Tabular Data
>
> Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

## Non-vendored packages, weights and datasets

Third-party Python packages and model weights are obtained separately and are
not redistributed in this repository. This includes the TabPFNv2 and TabICLv2
packages and weights. Real-world datasets are also not redistributed. Those
materials remain subject to their respective providers' licenses and terms.
