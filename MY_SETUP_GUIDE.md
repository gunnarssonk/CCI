# My Setup Guide for CCI Project

## Every time I open a new Terminal

```bash
# 1. Go to the project folder
cd "/Users/klara.gunnarsson/Library/CloudStorage/OneDrive-ESA/Documents/CODE/python-projects/CCI"

# 2. Activate the mamba environment (this is what gives me `uv`)
source setup/default/env.sh

# 3. Activate the project's virtual environment (torch, wandb, numpy, ...)
source .venv/bin/activate
```

The prompt should then show `(CCI) (esa_env)`. Both = ready to go.

Note on step 2: it has to be `source`, **not** `bash`. With `bash` the environment
switches itself off again the moment the script finishes.

## The two environments (they are different things!)

| | `esa_env` (mamba) | `.venv/` (in the project folder) |
|---|---|---|
| What it is | An environment I activate with `setup/default/env.sh` | A folder of installed packages |
| What's in it | basically just `uv` | torch, wandb, numpy, earthengine-api, rasterio... |
| Who made it | me, via `env.sh` | `uv`, automatically, the first time `data.sh`/`train.sh` ran |

Analogy: `env.sh` hands me the toolbox (`uv`); `.venv` is the tools inside it.
That's why `wandb` was "command not found" when only `esa_env` was active —
wandb was never installed there.

## Train the model

```bash
python -m src.train --epochs 5     # quick smoke test, ~5 passes over the data
python -m src.train                # the real thing (default 2000 epochs, hours)
```

An **epoch** = one full pass over all the training data. More epochs = better
results but longer runtime. `Ctrl + C` stops it any time.

Config defaults live in the `Config` dataclass in `src/train.py`. Any field there
can be overridden on the command line, e.g. `--batch_size 8 --lr 1e-3`.

## Weights & Biases (logging)

One time only:

```bash
wandb login
```

- Get the key from https://wandb.ai/authorize and use the **copy button** —
  selecting it by hand grabs extra text off the page.
- The key stays **invisible** while pasting. That's normal, just press Enter.
- A valid key is exactly **40 characters**. If login "succeeds" but training then
  dies with `CommError: user is not logged in`, the stored key is wrong →
  `wandb login --relogin`.

Don't want to use wandb at all? Type this each session instead:

```bash
export WANDB_MODE=offline
```

Runs then save to a local `wandb/` folder. Upload later with `wandb sync wandb/latest-run`.

## Downloading data

**I already have 100 samples in `data_uniform/`, so I normally skip this.**

```bash
python -m src.dataset.build
```

This fetches 100 *brand-new random* samples from France (AlphaEarth embeddings +
ESA biomass targets) and saves `.npy` files into `data_uniform/`. It does not check
what I already have — running it twice gives me 200 files. Slow and uses Earth
Engine quota.

## Where things live

| What | Where |
|------|-------|
| Project code | `~/Library/CloudStorage/OneDrive-ESA/Documents/CODE/python-projects/CCI` |
| Mamba environment | `~/miniforge3/envs/esa_env` (holds `uv`) |
| Python packages | `.venv/` inside the project folder |
| Training data | `data_uniform/ae_embeddings/` and `data_uniform/targets/` |
| wandb key | `~/.netrc` |
| Earth Engine key | `~/.config/earthengine/credentials` |

## Key commands I might forget

| What | Command |
|------|---------|
| List files in a folder | `ls` |
| Count my data samples | `ls data_uniform/ae_embeddings \| wc -l` |
| Install a Python package | `uv pip install <package-name>` |
| Stop something running | `Ctrl + C` |
| Pull latest code from GitHub | `git pull` |
| Leave an environment | `deactivate` (for `.venv`) / `mamba deactivate` (for `esa_env`) |

## If something breaks

**`import: command not found` / `syntax error near unexpected token` when running
`wandb` or `python`** → This is OneDrive. It doesn't understand symlinks, so when
it syncs the project it flattens `.venv/bin/python*` into plain text files, and the
shell can no longer find a real Python to run. Fix:

```bash
cd .venv/bin
rm python python3 python3.13
ln -s ~/miniforge3/bin/python3 python
ln -s python python3
ln -s python python3.13
./python -V          # should print Python 3.13.x
cd ../..
```

This will come back every time OneDrive re-syncs. The permanent fix is to move
`.venv` out of the OneDrive folder.

**`command not found: uv`** → I skipped `source setup/default/env.sh`.

**`ModuleNotFoundError`** → `.venv` isn't active (`source .venv/bin/activate`),
or the package is genuinely missing (`uv pip install <name>`).

**Earth Engine permission errors** → `earthengine authenticate`.

## Things I changed from the original setup

- **Google Cloud project**: changed `ee.Initialize(project="alexcloud-489214")` to
  `project="esa-cci"` in `src/dataset/examples.py` and `src/dataset/build.py`.
- **PyTorch**: `data.sh` installs the CUDA build (for Windows/Linux + NVIDIA).
  On Mac it just installs the CPU build, which is what training uses
  (`Using device: cpu` at startup).
- I previously made an env called `esa_env_312`; it's gone now. The one in use
  is `esa_env`, and the Python that actually matters is the 3.13 one inside `.venv`.
