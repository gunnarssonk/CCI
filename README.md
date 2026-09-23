# CCI — biomass from satellite embeddings

Can general-purpose satellite embeddings reconstruct an Essential Climate Variable?
This repo tests it on above-ground biomass (AGB): a small model reads Google
AlphaEarth embeddings and predicts ESA CCI biomass, pixel by pixel.

- **Input:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` — 64 channels per pixel, one image per year.
- **Target:** `projects/sat-io/open-datasets/ESA/ESA_CCI_AGB` — biomass in Mg/ha.
- **Tiles:** random points inside a country, each a ~10 km box downloaded as 256×256 pixels.
- **Models:** `PointWiseModel` (each pixel from its own 64 values) and `SmallCNN` (adds a 3×3 neighbourhood). A ridge regression serves as the linear baseline.

## Setup (once)

You need `mamba` (or `conda`). Check with `mamba --version`; if missing, install
[miniforge](https://github.com/conda-forge/miniforge).

```bash
source setup/default/env.sh    # Mac/Linux: creates + activates the mamba env `esa_env` (provides `uv`)
bash setup/default/data.sh     # installs packages into .venv/ via uv, authenticates Earth Engine
wandb login                    # once; key from https://wandb.ai/authorize (use the copy button)
```

Windows: `.\setup\windows\env.ps1`, `.\setup\windows\data.ps1` (edit the mamba path in `env.ps1` first).

There are **two environment layers**, and both are needed:

| | `esa_env` (mamba) | `.venv/` (in the project folder) |
|---|---|---|
| What it is | environment activated by `env.sh` | folder of installed packages |
| What's in it | basically just `uv` | torch, wandb, numpy, earthengine-api, rasterio, ... |
| Who made it | `env.sh` | `uv`, the first time `data.sh` ran |

`env.sh` hands you the toolbox (`uv`); `.venv` holds the tools. If `wandb` or `torch`
is "not found", `.venv` isn't active.

## Every session

```bash
cd <project folder>
source setup/default/env.sh    # must be `source`, not `bash` — bash runs it in a subshell and loses the activation
source .venv/bin/activate
```

The prompt shows `(CCI) (esa_env)` when both are active. Always run things as
modules from the project root (`python -m src.train`), never as file paths.

## Workflow

### 1. Download data

```bash
python -m src.dataset.build --country France --year 2020 --total_samples 100 --output_dir data_gee/data_france_2020_v2
```

Slow (~13 s per tile), uses Earth Engine quota. `--output_dir` is required and the
script **appends**: running twice into the same folder gives 200 tiles. Each folder
gets a `manifest.json` recording how it was built. Earth Engine project comes from
the `EE_PROJECT` env var (default `esa-cci`); re-auth with `earthengine authenticate`.

**Alternative: targets from the ESA Climate Toolbox** (any CCI ECV, not only biomass).
The toolbox ([docs](https://esa-climate-toolbox.readthedocs.io/), [GitHub](https://github.com/esa-cci/esa-climate-toolbox))
reads CCI datasets straight from the CCI Open Data Portal. It needs GDAL, which pip
cannot install into `.venv`, so it lives in its own conda env:

```bash
bash setup/default/ect_env.sh            # once
mamba activate ect                       # instead of .venv, for the two commands below
python -m src.dataset.ect_targets --ecv BIOMASS                   # list datasets; --list-ecvs for all ECVs
python -m src.dataset.build_ect --country France --year 2020 --total_samples 100 --output_dir data_gee/data_france_2020_ect
```

Embeddings still come from Earth Engine; only the target changes. Output layout is
identical, so `train.py` / `evaluate.py` (in `.venv`) work unchanged. Pass
`--data_id <dataset> --store <store>` for another ECV; value and uncertainty
variables are auto-detected or set with `--var` / `--sd_var`. The toolbox returns
data on its native grid; `ect_targets.py` resamples it (nearest neighbour) onto the
tile grid. `--backfill` / `--compare` fetch toolbox targets for an existing GEE dataset
and report pixel agreement.

Known limits (2026-09): the ODP store (`esa-cci`, AGB v5–v7) crashes inside the toolbox
for the AGB datasets, so the default is the Zarr store with AGB **v4** (2010–2020),
while the Earth Engine tiles are v6 — the two maps differ per pixel (correlation ≈0.9,
grid alignment verified). The tile approach needs the ECV's native resolution well
below the tile size (~20 km); coarse products (SST, soil moisture, 0.05–0.25°) are
rejected and would need a different design.

Current datasets (gitignored, one folder per region under `data_gee/`):

| Folder | What |
|---|---|
| `data_gee/data_france_2020_v2/` | France 2020, 100 tiles. Training set. |
| `data_gee/data_sweden_v2/` | Sweden 2020, 100 tiles. Out-of-domain test set, never trained on. |

### 2. Train

```bash
python -m src.train --epochs 5                                              # smoke test
python -m src.train --model_name SmallCNN --exp_name myexp --run_name cnn   # real run
```

Every field of `Config` in `src/train.py` is a CLI flag: `--epochs`, `--lr`,
`--batch_size`, `--patience`, `--data_dir`, `--normalize`, `--target_scale`, `--seed`, ...
Inputs are standardised per channel and targets divided by 100 (`target_scale`);
both are written to `train_meta.json` next to the checkpoint so evaluation can undo them.

- Checkpoints: `checkpoints/<exp_name>/<run_name>_<timestamp>/model.pth`, saved only on a new best val loss.
- Early stopping after `--patience` epochs (default 50) without improvement.
- Logs to Weights & Biases as `<exp_name>_<run_name>_<dataset>_<timestamp>`.
  `export WANDB_MODE=offline` to skip the account; `wandb sync wandb/latest-run` uploads later.
- Train/val/test split is by `--seed` (default 42) over sorted filenames, so it is the same in every run.

Long runs from a terminal you want to close:

```bash
nohup python -m src.train ... > logs/myrun.txt 2>&1 &
grep -o "Epoch [0-9]*/600 | Train Loss.*" logs/myrun.txt | tail -1   # progress
```

### 3. Evaluate

```bash
# a checkpoint on the held-out test tiles of its own dataset
python -m src.evaluate --checkpoint checkpoints/myexp/cnn_.../model.pth --model_name SmallCNN

# the same checkpoint on another region (all tiles, since none were trained on)
python -m src.evaluate --checkpoint ... --model_name SmallCNN --data_dir data_gee/data_sweden_v2 --split all --name cnn_on_sweden

# linear baseline: ridge fitted on the train split, scored on the test split
python -m src.evaluate --model_name Ridge
```

Writes `results/<name>/` with `metrics.json` (RMSE, MAE, R², bias, per-tile spread),
`per_tile.csv`, `scatter.png` and a few example tile figures. Only valid pixels count.

CCI ships a per-pixel standard deviation (its own uncertainty). `build.py` saves it as
`targets_sd/`; for older datasets back-fill it with
`python -m src.dataset.fetch_sd --data_dir <dir>`. When present, `evaluate.py` also
reports the model error relative to it (`rmse_over_cci_sd`, `frac_within_1sd`) and
plots `error_vs_cci_sd.png`. A ratio near 1 means the model is as accurate as the
reference map claims to be.

## Results so far

France 2020, 15 held-out test tiles; Sweden, 100 tiles never trained on:

| Model | France RMSE (Mg/ha) | France R² | Sweden RMSE | Sweden R² |
|---|---|---|---|---|
| Ridge (linear) | 31.7 | 0.73 | – | – |
| PointWise | 27.7 | 0.80 | 28.9 | 0.60 |
| SmallCNN | 26.3 | 0.82 | 27.0 | 0.65 |

A linear model already explains ~70 % of the variance, so most of the biomass
signal is present in the embeddings directly. Results are stable across two
independent random samples of France. Transfer to Sweden keeps R² at 0.6–0.7 but
the mean bias varies between training runs (−13 to +10 Mg/ha); seed repeats are
needed before interpreting it.

## Layout

| Path | What |
|---|---|
| `src/dataset/build.py` | Earth Engine download loop → `.npy` tiles + manifest |
| `src/dataset/dataset.py` | `BiomassDataset`: tile loading, no-data masking, normalisation, seeded split |
| `src/model/model.py` | `SmallCNN`, `PointWiseModel` |
| `src/train.py` | `Config`, training loop, early stopping, wandb |
| `src/evaluate.py` | Metrics, plots, ridge baseline, comparison to CCI uncertainty |
| `src/dataset/ect_targets.py` | ESA Climate Toolbox access + regridding; discovery / back-fill / compare CLI (env `ect`) |
| `src/dataset/build_ect.py` | Dataset builder with toolbox targets, same layout as `build.py` (env `ect`) |
| `src/dataset/fetch_sd.py` | Back-fill CCI uncertainty for datasets built before `build.py` saved it |
| `setup/` | Environment scripts for Mac/Linux (`default/`) and Windows; `ect_env.sh` for the toolbox env |
| `data_gee/`, `checkpoints/`, `results/`, `logs/`, `wandb/`, `visu/` | Local outputs, gitignored except `results/` if you choose to commit it |

## Troubleshooting

**`import: command not found` or `syntax error near unexpected token` from anything in `.venv/bin`**
— OneDrive. It flattens the venv's `python*` symlinks into text files on sync. Repair:

```bash
BASE=$(awk -F'= *' '/^home/{print $2}' .venv/pyvenv.cfg)/python3   # interpreter the venv was built from
cd .venv/bin && rm python python3 python3.13
ln -s "$BASE" python && ln -s python python3 && ln -s python python3.13
./python -V && cd ../..
```

Recurs after every re-sync; the durable fix is keeping `.venv` outside the synced folder.

**`command not found: uv`** — `env.sh` wasn't sourced.
**`ModuleNotFoundError`** — `.venv` isn't active, or `uv pip install <name>`.
**`CommError: user is not logged in`** — stored wandb key is wrong; `wandb login --relogin`. A valid key is exactly 40 characters.
**Earth Engine permission errors** — `earthengine authenticate`.
**`OMP: Error #15` at startup** — two training processes launched within a couple of seconds of each other; relaunch the one that died.
**`No *_ae.npy tiles found`** — wrong `--data_dir`.

## Handy commands

| What | Command |
|---|---|
| Count tiles in a dataset | `ls data_gee/data_france_2020_v2/ae_embeddings \| wc -l` |
| Install a package | `uv pip install <name>` |
| Leave the environments | `deactivate`, then `mamba deactivate` |
| Stop a foreground run | `Ctrl + C` |
| Stop a background run | `pkill -f "run_name cnn"` |

## Open ideas (raw notes)

- Year alignment at test time; how AlphaEarth's annual embedding relates to the CCI temporal mosaic (biomass is seasonal, CCI averages).
- Geographic sampling: keep whole zones/continents for testing; biome representativity; clouds.
- Compare the error to CCI's own per-pixel uncertainty layer; use it as loss weights.
- Correlation between biomass and individual embedding channels / PCA.
- Distribution-oriented scatter; low- vs high-biomass diversity; keep training inputs out of test visuals.
- Major TOM (Hugging Face) as another embedding source; GeoTessera as an alternative to AlphaEarth.
