
# Setup
## Environment setup
first test if you already have mamba installed, by running ` mamba --version ` <br>
if not check conda :   ` conda --version ` if it's there, do  ` conda install mamba -n base -c conda-forge ` <br>
if no conda detected, follow the instructions of https://github.com/conda-forge/miniforge#mambaforge <br>

Change in the 'env.ps1' file your Conda/Mamba path, then in command line :  <br>
windows : run  ` .\setup\windows\env.ps1 `          <br>
apple/linux : run `source setup/default/env.sh `    <br>
log in on Earth Engine following the instructions.  

Use `source`, not `bash`, for `env.sh` — `bash` runs it in a subshell, so the
environment is deactivated again as soon as the script exits.

## Dataset building
windows : run ` .\setup\windows\data.ps1 `  
apple/linux : run `bash .\setup\default\data.sh `

### Data stockage:
Google EarthEngine / GeoTessera  *(run once, slow, quota-limited)*  
  $\to$ .npy files on disk       *(cheap, permanent, version-controlled)*  
  $\to$ mmap'd Dataset           *(zero RAM overhead for tile storage)*  
  $\to$ patch sampler            *(random crops → huge effective dataset size)*  
  $\to$ DataLoader               *(transfer to the model/task during training)*

(once we run out data of our initial fecthing to train on, we can pull fresh data from new zones)

Note: `setup/default/shared_setup.sh` is not run directly — `data.sh` and `train.sh` both load it
automatically to share their common install/auth steps.

# Model training
windows : run ` .\setup\windows\train.ps1 `  
apple/linux : run `bash .\setup\default\train.sh `

Training logs to Weights & Biases, so run `wandb login` once beforehand (key from
https://wandb.ai/authorize — use the copy button, and note the key stays invisible
while pasting). To run without an account, `export WANDB_MODE=offline` instead.

The setup scripts install everything into a `.venv/` folder via `uv`. Once that
exists you can skip the scripts and run the steps directly:

```bash
source setup/default/env.sh   # mamba env, provides uv
source .venv/bin/activate     # the packages themselves
python -m src.train --epochs 5   # short smoke test; drop the flag for the full run
python -m src.dataset.build      # only if you want more data
```

Every field of the `Config` dataclass in `src/train.py` is exposed as a CLI flag
(`--epochs`, `--batch_size`, `--lr`, `--patch_size`, ...).


# Troubleshooting
**`import: command not found` or `syntax error near unexpected token` when running
anything from `.venv/bin`** — if the repo lives in a OneDrive/Dropbox-synced folder,
the sync client flattens the venv's symlinks into plain text files, leaving no
working Python for the shebang to find. Repair with:

```bash
# the interpreter this venv was built from is recorded in .venv/pyvenv.cfg ("home =")
BASE=$(awk -F'= *' '/^home/{print $2}' .venv/pyvenv.cfg)/python3
cd .venv/bin && rm python python3 python3.13
ln -s "$BASE" python && ln -s python python3 && ln -s python python3.13
./python -V   # sanity check
```

It will recur on every re-sync; the durable fix is to keep `.venv` outside the
synced folder.

**`command not found: uv`** — `env.sh` wasn't sourced.


# notes
year alignment (test-time usage)
directions :
alphaEarth combines more granular (monthly) division
geographical, random sampling in a defined zones, keep some zones (continent) for testing
clouds, biomes representativity

considers temporal info, biomass how temporal mosaic been considered (other dataset? land cover is snapshot)
biomass depends season, cci averaging
how data temporally in AlphaEarth
distribution oriented scatterplot

reconstructed and ground truth
don't want train input in the test visuals, few locations used as visualizations place (?)
diversified low biomass, high biomass

std average constraints


dataset map finishing
correlation between biomass and each of the biomass and embeddings (pca)
simples models 

major tom huggingface