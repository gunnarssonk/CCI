# What the columns in `all_results.csv` mean

One row = one evaluation (one model scored on one dataset).
The table is rebuilt by `python -m src.collect_results`; do not edit it by hand.

| Column | Plain meaning |
|---|---|
| `name` | The results folder this row comes from: `results/<name>/`. |
| `model` | Which kind of model. `SmallCNN` looks at neighbouring pixels, `PointWiseModel` looks at one pixel at a time, `Ridge` is the simple straight-line baseline. `x3` means three models averaged (an ensemble). |
| `trained_on` | The data folder the model learned from. A `+` between two names means it learned from both together. |
| `n_train_tiles` | How many tiles it learned from. |
| `init_seed` | The number that fixes the model's random starting point. Rows that differ only in this number are repeat runs of the same setup. |
| `scored_on` | The data folder the model was tested on. |
| `split` | Which tiles of that folder were used. `test` = the 15 % of tiles kept away from training. `all` = every tile, used when the model never saw that folder (transfer to a new region). |
| `rmse` | Typical size of the error, in Mg/ha. Big misses count extra. Lower is better. |
| `mae` | Average miss, in Mg/ha. Lower is better. |
| `r2` | How much of the variation in biomass the model explains. 1 = perfect. 0 = no better than saying "the average" everywhere. Below 0 = worse than that. |
| `bias` | Prediction minus truth, averaged, in Mg/ha. Above 0 = the model predicts too much. Below 0 = too little. |
| `rmse_over_cci_sd` | Our error divided by the uncertainty the CCI map gives for itself. Below 1 = we disagree with the map by less than the map is unsure of itself. |
| `frac_within_1sd` | Share of pixels where our miss is smaller than the map's own uncertainty for that pixel. 0.69 means 69 %. Higher is better. |
| `date` | The day the evaluation was run. |
| `checkpoint_path` | Where the saved model file is. Empty for `Ridge`, which has no saved model. |

## Reading the cells

- **Empty cell** = it was never written down. Nothing is guessed.
  - `n_train_tiles` is empty for every model trained before 2026-11-12.
  - `trained_on` and `init_seed` are empty when the model's folder is not on the computer where the table was built. Build the table on JASMIN to fill most of them.
  - `rmse_over_cci_sd` and `frac_within_1sd` are empty when the dataset has no uncertainty files (`targets_sd/`).
- **`;` inside a cell** = one value per member of an ensemble, in order. `;1;2` means: first member not written down, then seed 1, then seed 2.
- **Old folder name:** `data_france_2020_j` is the same data as `data_france_2020_j300` (renamed 2026-11-10).
