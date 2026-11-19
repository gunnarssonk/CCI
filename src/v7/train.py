"""Train one model on v7 blocks, with exactly the v6 training code and settings.

    python -m src.v7.train --exp all6 --run_name cnn --model_name SmallCNN
    python -m src.v7.train --exp nonl --run_name pw --model_name PointWiseModel \
        --boxes central_sweden,northern_sweden,southwest_sweden,amapa_brazil,east_para_brazil

(The experiments are normally started through src/v7/experiments.py, which calls run() below.)

This is a thin wrapper around src/train.py: the model, the loss, the learning rate,
the batch size, early stopping and the normalisation are the ones in src/train.py, untouched.
The wrapper only decides where things go and what data is used:

  data         <DATA_ROOT>/blocks/<box>/ for the chosen boxes (one folder per box)
  split        the saved block split (make_split.py); checked against what the data
               loader actually produces before training starts
  model        checkpoints/v7/<exp>/<run_name>_<date>_<time>/model.pth  (+ train_meta.json)
  settings     run_train_*.json next to it: the exact command and every setting
  pictures     visu/v7/<exp>_<run_name>/   (the v6 pictures in visu/ are left alone)
  wandb files  logs/v7/wandb/              (offline, as in v6)

Settings that are not flags here keep the v6 values: batch 16, learning rate 1e-4,
patience 50, inputs standardised, targets divided by 100, split seed 42, up to 2000 epochs.

It refuses to train if any block of the chosen boxes holds left-out pixels (the no-data
marker): the v6 code that works out the input statistics would read the marker as a value.
"""
import os, json, glob, argparse
from dataclasses import asdict

import numpy as np

from src.v7.common import BOXES, DATA_ROOT, write_run_record


def run(exp, run_name, model_name="SmallCNN", boxes=None, init_seed=None, epochs=2000):
    boxes = list(boxes or BOXES)
    for b in boxes:
        if b not in BOXES:
            raise SystemExit(f"unknown box {b!r}; choose from {', '.join(BOXES)}")
    data_dirs = [os.path.join(DATA_ROOT, "blocks", b) for b in boxes]
    left_out = 0
    for d in data_dirs:
        files = glob.glob(os.path.join(d, "valid", "*_valid.npy"))
        if len(files) != 25 or len(glob.glob(os.path.join(d, "ae_embeddings", "*_ae.npy"))) != 25:
            raise SystemExit(f"{d}: {len(files)} blocks found, expected 25; run src.v7.build_blocks first")
        left_out += sum(int((np.load(f) < 0.999).sum()) for f in files)
    if left_out:
        raise SystemExit(f"{left_out} pixels in these boxes are left out for missing embeddings. The v6 code that works out the "
                         "input statistics would read the no-data marker as a value, so training is not started. Tell Klara.")

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", os.path.join("logs", "v7"))
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    from src import train as T                      # the v6 training code, unchanged
    from src.dataset.dataset import BiomassDataset

    cfg = T.Config(model_name=model_name, data_dir=",".join(data_dirs), exp_name=f"v7/{exp}",
                   run_name=run_name, epochs=epochs, init_seed=init_seed)

    # ---- the split the data loader will produce must be the saved block split ----
    with open(os.path.join(DATA_ROOT, "split_v7_seed42.json")) as f:
        saved = json.load(f)
    assert saved["seed"] == cfg.seed
    ratio = (cfg.split_train, cfg.split_val, 1.0 - cfg.split_train - cfg.split_val)
    counts = {}
    for split in ("train", "val", "test"):
        ds = BiomassDataset(cfg.data_dir, split=split, split_ratio=ratio, use_ae=cfg.use_ae, seed=cfg.seed)
        got = sorted(name for _, _, name in ds.samples)
        want = sorted(n for b in boxes for n in saved["boxes"][b][split])
        if got != want:
            raise SystemExit(f"the {split} blocks from the data loader differ from the saved split file; not training")
        counts[split] = len(got)
    print(f"split check passed: the data loader gives exactly the saved split for {len(boxes)} box(es): {counts}")

    # ---- pictures go to visu/v7/..., not into the v6 visu/ folder ----
    visu_dir = os.path.join("visu", "v7", f"{exp}_{run_name}")
    original = T.visualize_predictions
    T.visualize_predictions = lambda x, y_true, y_pred, epoch, save_dir=None: original(x, y_true, y_pred, epoch, save_dir=visu_dir)

    exp_dir = os.path.join("checkpoints", "v7", exp)
    os.makedirs(exp_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(exp_dir, "*")))
    settings = {"exp": exp, "run_name": run_name, "model_name": model_name, "boxes": boxes, "blocks": counts,
                "training_config": asdict(cfg), "split_file": os.path.join(DATA_ROOT, "split_v7_seed42.json"),
                "pictures": visu_dir, "wandb_dir": os.environ["WANDB_DIR"], "training_code": "src/train.py (v6, unchanged)"}
    write_run_record(exp_dir, f"train_{run_name}_started", settings=settings)

    T.train(cfg=cfg)

    new = sorted(d for d in set(glob.glob(os.path.join(exp_dir, f"{run_name}_*"))) - before if os.path.isdir(d))
    run_dir = new[-1] if new else exp_dir
    write_run_record(run_dir, "train", settings=settings, results={"model": os.path.join(run_dir, "model.pth")})
    print(f"model -> {run_dir}/model.pth")
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="experiment name, e.g. all6; the model goes to checkpoints/v7/<exp>/")
    p.add_argument("--run_name", required=True, help="e.g. cnn, pw, cnn_s1")
    p.add_argument("--model_name", default="SmallCNN", choices=["SmallCNN", "PointWiseModel"])
    p.add_argument("--boxes", default=",".join(BOXES), help="comma-separated boxes to train on; default all six")
    p.add_argument("--init_seed", type=int, default=None, help="only for repeat runs; the split seed stays 42")
    p.add_argument("--epochs", type=int, default=2000, help="upper limit; early stopping ends the run (2000 as in the v6 job script)")
    a = p.parse_args()
    run(a.exp, a.run_name, a.model_name, [b.strip() for b in a.boxes.split(",") if b.strip()], a.init_seed, a.epochs)


if __name__ == "__main__":
    main()
