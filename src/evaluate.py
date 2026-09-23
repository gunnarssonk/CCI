"""Evaluate a trained checkpoint (or a ridge-regression baseline) on a dataset split.

Produces the numbers a feasibility study needs: RMSE, MAE, R², bias, on valid
pixels only, plus a scatter plot and a per-tile CSV. Everything lands in
results/<name>/.

Examples
--------
# trained model on the held-out test split of the dataset it was trained on
python -m src.evaluate --checkpoint checkpoints/france2020_long/cnn_0907_1024/model.pth \
    --model_name SmallCNN --data_dir data_gee/data_france_2020

# same checkpoint on a different region (all tiles, since none were used for training)
python -m src.evaluate --checkpoint checkpoints/france2020_long/cnn_0907_1024/model.pth \
    --model_name SmallCNN --data_dir data_gee/data_sweden --split all --name cnn_on_sweden

# ridge baseline: fit on the train split, evaluate on the test split
python -m src.evaluate --model_name Ridge --data_dir data_gee/data_france_2020
"""
import os, json, csv, argparse
from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset.dataset import BiomassDataset
from src.model.model import SmallCNN, PointWiseModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"): return True
    if v.lower() in ("false", "f", "no", "n", "0"): return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


MODELS = {
    "PointWiseModel": PointWiseModel,
    "SmallCNN": SmallCNN,
}


@dataclass
class Config:
    model_name: str = "PointWiseModel"   # one of MODELS, or "Ridge"
    checkpoint: str = None               # path to model.pth; not needed for Ridge
    data_dir: str = "data_gee/data_france_2020_v2"
    split: str = "test"                  # train | val | test | all
    name: str = None                     # results/<name>/; defaults to checkpoint folder name
    out_root: str = "results"

    # must match the training run so the split is the same
    seed: int = 42
    split_train: float = 0.7
    split_val: float = 0.15

    batch_size: int = 8
    ridge_alpha: float = 1.0
    ridge_max_pixels: int = 500_000      # subsample of train pixels used to fit Ridge
    example_tiles: int = 3               # how many tile figures to save


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_split(cfg, split, norm_stats=None):
    ratio = (cfg.split_train, cfg.split_val, 1.0 - cfg.split_train - cfg.split_val)
    kw = dict(split_ratio=ratio, use_ae=True, seed=cfg.seed, norm_stats=norm_stats)
    if split == "all":
        ds = BiomassDataset(cfg.data_dir, split="train", **kw)
        for s in ("val", "test"):
            ds.samples += BiomassDataset(cfg.data_dir, split=s, **kw).samples
        return ds
    return BiomassDataset(cfg.data_dir, split=split, **kw)


def load_train_meta(checkpoint):
    """train.py writes train_meta.json next to model.pth (since 2026-09-23) with the
    input stats and target scale the model was trained with. Older checkpoints have
    none: they were trained on raw inputs and raw Mg/ha targets."""
    path = os.path.join(os.path.dirname(checkpoint), "train_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"normalize": False, "norm_stats": None, "target_scale": 1.0}


def target_mask(y):
    """Valid target pixels: finite and non-negative. CCI AGB no-data can show up as NaN
    or a negative fill value depending on how the tile was exported."""
    return torch.isfinite(y) & (y >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# Predictors: both return (pred, y, valid, name) per batch, on CPU
# ─────────────────────────────────────────────────────────────────────────────
def predict_torch(model, loader, device, target_scale=1.0):
    model.eval()
    with torch.no_grad():
        for x, y, valid, names in loader:
            y = y.squeeze(-1)
            pred = model(x.to(device)).cpu() * target_scale   # back to Mg/ha
            valid = valid.bool() & target_mask(y)
            yield pred, y, valid, names


def fit_ridge(cfg):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_ds = load_split(cfg, "train")
    rng = np.random.default_rng(cfg.seed)
    per_tile = max(1, cfg.ridge_max_pixels // len(train_ds))
    xs, ys = [], []
    for i in range(len(train_ds)):
        x, y, valid, _ = train_ds[i]
        y = y.squeeze(-1)
        ok = (valid.bool() & target_mask(y)).numpy().reshape(-1)
        xf = x.numpy().reshape(-1, x.shape[-1])[ok]
        yf = y.numpy().reshape(-1)[ok]
        if len(yf) > per_tile:
            idx = rng.choice(len(yf), per_tile, replace=False)
            xf, yf = xf[idx], yf[idx]
        xs.append(xf); ys.append(yf)
    X = np.concatenate(xs); Y = np.concatenate(ys)
    print(f"Ridge: fitting on {len(Y):,} pixels from {len(train_ds)} train tiles")
    model = make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
    model.fit(X, Y)
    return model


def predict_ridge(model, loader):
    for x, y, valid, names in loader:
        y = y.squeeze(-1)
        B, H, W, C = x.shape
        flat = x.numpy().reshape(-1, C)
        pred = torch.from_numpy(model.predict(flat).reshape(B, H, W)).float()
        valid = valid.bool() & target_mask(y)
        yield pred, y, valid, names


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(pred, true):
    """All inputs 1-D numpy arrays of valid pixels."""
    err = pred - true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return {
        "n_pixels": int(len(true)),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),                  # >0 means over-prediction
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "true_mean": float(true.mean()),
        "true_std": float(true.std()),
        "pred_mean": float(pred.mean()),
    }


def sd_metrics(pred, true, sd):
    """Compare the model's error to CCI's own per-pixel uncertainty.

    CCI reports SD = 0 wherever AGB = 0 (no forest, nothing to be uncertain
    about), so the "within k·SD" fractions are computed on SD > 0 pixels only.
    cci_sd_rms is sqrt(mean(SD²)), the number directly comparable to RMSE: if the
    reference itself has that much noise, a model cannot be expected to beat it.
    """
    err = np.abs(pred - true)
    has_sd = sd > 0
    out = {
        "cci_sd_mean": float(sd.mean()),
        "cci_sd_rms": float(np.sqrt((sd ** 2).mean())),
        "frac_sd_gt0": float(has_sd.mean()),
    }
    out["rmse_over_cci_sd"] = float(np.sqrt(((pred - true) ** 2).mean()) / out["cci_sd_rms"]) if out["cci_sd_rms"] > 0 else float("nan")
    if has_sd.any():
        out["frac_within_1sd"] = float((err[has_sd] <= sd[has_sd]).mean())
        out["frac_within_2sd"] = float((err[has_sd] <= 2 * sd[has_sd]).mean())
    else:
        out["frac_within_1sd"] = out["frac_within_2sd"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def error_vs_sd_plot(pred, true, sd, m, title, path):
    """Model |error| binned by CCI SD. Points on the 1:1 line mean the model is
    exactly as wrong as CCI says the reference is uncertain."""
    err = np.abs(pred - true)
    ok = sd > 0
    if ok.sum() < 100:
        return
    edges = np.percentile(sd[ok], np.linspace(0, 100, 21))
    edges = np.unique(edges)
    idx = np.digitize(sd[ok], edges[1:-1])
    xs, med, p25, p75, rms = [], [], [], [], []
    for i in range(len(edges) - 1):
        sel = idx == i
        if sel.sum() < 50:
            continue
        xs.append(sd[ok][sel].mean())
        e = err[ok][sel]
        med.append(np.median(e)); p25.append(np.percentile(e, 25)); p75.append(np.percentile(e, 75))
        rms.append(np.sqrt((e ** 2).mean()))
    xs, med, p25, p75, rms = map(np.array, (xs, med, p25, p75, rms))
    hi = float(max(xs.max(), rms.max()) * 1.05)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(xs, p25, p75, alpha=0.25, label="model |error|, 25–75 %")
    ax.plot(xs, med, "-o", ms=4, label="model |error|, median")
    ax.plot(xs, rms, "-s", ms=4, label="model error, RMS")
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1 (error = CCI SD)")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("CCI reported SD (Mg/ha), binned")
    ax.set_ylabel("Model error (Mg/ha)")
    ax.set_title(title)
    ax.text(0.03, 0.97,
            f"RMSE {m['rmse']:.1f}   CCI SD (rms) {m['cci_sd_rms']:.1f}\nratio {m['rmse_over_cci_sd']:.2f}\n"
            f"within 1 SD: {100 * m['frac_within_1sd']:.0f} %   within 2 SD: {100 * m['frac_within_2sd']:.0f} %",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def scatter_plot(pred, true, m, title, path, max_points=300_000):
    if len(true) > max_points:
        idx = np.random.default_rng(0).choice(len(true), max_points, replace=False)
        pred, true = pred[idx], true[idx]
    hi = float(np.percentile(np.concatenate([pred, true]), 99.5))
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(true, pred, gridsize=80, extent=(0, hi, 0, hi), bins="log", cmap="viridis", mincnt=1)
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("ESA CCI biomass (Mg/ha)")
    ax.set_ylabel("Predicted biomass (Mg/ha)")
    ax.set_title(title)
    ax.text(0.03, 0.97,
            f"RMSE {m['rmse']:.1f}  MAE {m['mae']:.1f}\nR² {m['r2']:.3f}  bias {m['bias']:+.1f}\nn = {m['n_pixels']:,}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.colorbar(hb, ax=ax, label="pixels (log)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def tile_figure(x, y, pred, valid, name, rmse, path):
    vmax = float(np.nanpercentile(y[valid], 99)) if valid.any() else 1.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # no-data pixels were zeroed by the dataset; hide them here too so they don't
    # render as a flat block that looks like a bug
    ax[0].imshow(np.where(valid, x.mean(-1), np.nan), cmap="viridis"); ax[0].set_title("Embedding (64-ch mean)")
    im = ax[1].imshow(np.where(valid, y, np.nan), cmap="YlGn", vmin=0, vmax=vmax); ax[1].set_title("ESA CCI biomass")
    plt.colorbar(im, ax=ax[1], fraction=0.046, label="Mg/ha")
    im = ax[2].imshow(np.where(valid, pred, np.nan), cmap="YlGn", vmin=0, vmax=vmax); ax[2].set_title("Prediction")
    plt.colorbar(im, ax=ax[2], fraction=0.046, label="Mg/ha")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{name}  ·  RMSE {rmse:.1f} Mg/ha")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_stats = None
    if cfg.model_name == "Ridge":
        name = cfg.name or f"ridge_{os.path.basename(cfg.data_dir.rstrip('/'))}_{cfg.split}"
        model = fit_ridge(cfg)
        make_batches = lambda loader: predict_ridge(model, loader)
    else:
        if cfg.model_name not in MODELS:
            raise ValueError(f"Unknown model_name {cfg.model_name!r}. Choose one of: {', '.join(MODELS)}, Ridge")
        if not cfg.checkpoint:
            raise ValueError("--checkpoint is required for a trained model")
        ckpt_dir = os.path.basename(os.path.dirname(cfg.checkpoint))
        name = cfg.name or f"{ckpt_dir}_{os.path.basename(cfg.data_dir.rstrip('/'))}_{cfg.split}"

        meta = load_train_meta(cfg.checkpoint)
        norm_stats = meta["norm_stats"] if meta.get("normalize") else None
        target_scale = float(meta.get("target_scale", 1.0))
        print(f"Checkpoint meta: normalize={norm_stats is not None}, target_scale={target_scale}")

        eval_ds_probe = load_split(cfg, cfg.split, norm_stats)
        in_channels = eval_ds_probe[0][0].shape[-1]
        model = MODELS[cfg.model_name](in_channels).to(device)
        model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
        make_batches = lambda loader: predict_torch(model, loader, device, target_scale)

    out_dir = os.path.join(cfg.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    tiles_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    eval_ds = load_split(cfg, cfg.split, norm_stats)
    loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {cfg.model_name} on {cfg.data_dir} [{cfg.split}]: {len(eval_ds)} tiles")

    # CCI's own per-pixel uncertainty (SD band), if it was downloaded for this dataset
    sd_dir = os.path.join(cfg.data_dir, "targets_sd")
    have_sd = os.path.isdir(sd_dir)

    all_pred, all_true, all_sd, per_tile = [], [], [], []
    saved_examples = 0
    tile_idx = 0   # position in eval_ds, since the loader is not shuffled
    for pred, y, valid, names in make_batches(loader):
        for b in range(pred.shape[0]):
            v = valid[b].numpy()
            p = pred[b].numpy(); t = y[b].numpy()
            this_idx = tile_idx; tile_idx += 1
            if v.sum() == 0:
                print(f"  skipping {names[b]}: no valid pixels")
                continue
            m = metrics(p[v], t[v])
            sd_path = os.path.join(sd_dir, f"{names[b]}_sd.npy")
            if have_sd and os.path.exists(sd_path):
                sd = np.load(sd_path).squeeze()
                m.update(sd_metrics(p[v], t[v], sd[v]))
                all_sd.append(sd[v])
            per_tile.append({"tile": names[b], **m})
            all_pred.append(p[v]); all_true.append(t[v])
            if saved_examples < cfg.example_tiles:
                x_np = eval_ds[this_idx][0].numpy()
                tile_figure(x_np, t, p, v, names[b], m["rmse"], os.path.join(tiles_dir, f"{names[b]}.png"))
                saved_examples += 1

    all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)
    overall = metrics(all_pred, all_true)
    tile_rmse = np.array([r["rmse"] for r in per_tile])
    overall["tile_rmse_mean"] = float(tile_rmse.mean())
    overall["tile_rmse_std"] = float(tile_rmse.std())
    overall["n_tiles"] = len(per_tile)

    if all_sd and len(all_sd) == len(per_tile):
        all_sd = np.concatenate(all_sd)
        overall.update(sd_metrics(all_pred, all_true, all_sd))
        error_vs_sd_plot(all_pred, all_true, all_sd, overall,
                         f"{cfg.model_name} · {os.path.basename(cfg.data_dir.rstrip('/'))} [{cfg.split}]",
                         os.path.join(out_dir, "error_vs_cci_sd.png"))
    elif all_sd:
        print(f"  SD found for only {len(all_sd)} of {len(per_tile)} tiles; skipping the SD comparison")

    # ---- write ----
    summary = {
        "name": name,
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
        "metrics": overall,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "per_tile.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile[0].keys()))
        w.writeheader(); w.writerows(sorted(per_tile, key=lambda r: r["rmse"]))
    scatter_plot(all_pred, all_true, overall,
                 f"{cfg.model_name} · {os.path.basename(cfg.data_dir.rstrip('/'))} [{cfg.split}]",
                 os.path.join(out_dir, "scatter.png"))

    print()
    print(f"{'metric':<16}{'value':>12}")
    keys = ["rmse", "mae", "bias", "r2", "tile_rmse_mean", "tile_rmse_std", "true_mean", "true_std", "n_tiles", "n_pixels"]
    if "cci_sd_rms" in overall:
        keys += ["cci_sd_mean", "cci_sd_rms", "rmse_over_cci_sd", "frac_within_1sd", "frac_within_2sd"]
    for k in keys:
        v = overall[k]
        print(f"{k:<16}{v:>12,.3f}" if isinstance(v, float) else f"{k:<16}{v:>12,}")
    print(f"\nWritten to {out_dir}/")
    return overall


def args_extract(parser):
    for field in fields(Config):
        t = str2bool if field.type is bool else field.type
        parser.add_argument(f"--{field.name}", type=t, default=field.default)
    return vars(parser.parse_args())


if __name__ == "__main__":
    cfg = Config(**args_extract(argparse.ArgumentParser()))
    evaluate(cfg)
