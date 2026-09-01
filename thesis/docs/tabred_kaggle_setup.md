# TabReD source-file setup

This repository does not distribute the TabReD source data. Obtain each dataset
from the source named below and comply with its current terms. Place the
downloaded files in the default directory or pass another directory with
`--raw-dir`.

| Dataset | Source identifier | Default raw directory | Principal expected files |
| --- | --- | --- | --- |
| Cooking Time | `pcovkrd84mejm/cooking-time` | `data/tabred/raw/cooking-time/` | `cooking_time.parquet` |
| Delivery ETA | `pcovkrd84mejm/delivery-eta` | `data/tabred/raw/delivery-eta/` | `delivery_eta.parquet` |
| Ecom Offers | `acquire-valued-shoppers-challenge` | `data/tabred/raw/ecom-offers/` | `offers.csv.gz`, `trainHistory.csv.gz`, `transactions.csv.gz` |
| Homesite | `homesite-quote-conversion` | `data/tabred/raw/homesite/` | `train.csv.zip` or its extracted CSV |
| Sberbank | `sberbank-russian-housing-market` | `data/tabred/raw/sberbank/` | `train.csv.zip`, `macro.csv.zip` and `BAD_ADDRESS_FIX.xlsx` from the upstream TabReD preprocessing resources |
| Weather | `pcovkrd84mejm/tabred-weather` | `data/tabred/raw/weather/` | `weather.parquet` |
| Home Credit | `home-credit-credit-risk-model-stability` | `data/tabred/raw/homecredit/` | The parquet tree listed in `scripts/preprocessing/tabred_homecredit.py` |
| Maps Routing | `pcovkrd84mejm/maps-routing` | `data/tabred/raw/maps-routing/` | `maps_routing.parquet` |

The reported study uses the first six datasets in this table. Home Credit and
Maps Routing preprocessing code is retained for completeness, but
those two datasets are not part of the reported experimental scope.

Run a preprocessor from the repository root. For example:

```bash
python scripts/preprocessing/tabred_weather.py --overwrite
```

Each script writes `train.parquet`, `test.parquet` and `meta.json` under
`data/datasets/tabred_<name>/`. The source timestamp is retained on disk for
constructing the temporal split and is removed before the data reach a
prediction model.

After preparing the four datasets evaluated with different training periods,
construct the additional versions with:

```bash
python scripts/characterize_gap_widening.py
python scripts/build_gap_widened_tabred.py
```

See `DATASETS.md` at the repository root for the complete public data boundary.
