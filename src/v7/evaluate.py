"""Step 7 of the v7 setup: score a model against CCI biomass on the test blocks.

    # a trained model (the boxes it trained on are read from its train_meta.json)
    python -m src.v7.evaluate --name all6_cnn --model_name SmallCNN --checkpoint checkpoints/v7/all6/cnn_<stamp>/model.pth
    # the same model on boxes it never saw
    python -m src.v7.evaluate --name without_netherlands_cnn --model_name SmallCNN --checkpoint ... --score_boxes netherlands
    # baselines (no checkpoint): fitted here on the train blocks of --train_boxes
    python -m src.v7.evaluate --name all6_ridge --model_name Ridge
    python -m src.v7.evaluate --name all6_constant_mean --model_name ConstantMean
    python -m src.v7.evaluate --name all6_box_mean --model_name BoxMean

Models: SmallCNN and PointWiseModel (the v6 models), Ridge (the v6 linear baseline),
ConstantMean (always the mean biomass of the train blocks), BoxMean (the mean biomass of
the train blocks of the same box; a stricter baseline, because telling Sweden from Brazil
is easy and already gives a high overall R²).

Numbers: RMSE, bias (prediction minus CCI), R², MAE and RMSE divided by CCI's own SD,
  - over all scored boxes, per box, and per CCI biomass range
    (0-50, 50-100, 100-150, 150-200, 200-250, 250-300, 300-400, >400 Mg/ha),
  - on the test blocks (--pile test, the default), or on all 25 blocks of each box
    (--pile all; refused unless the model never trained on any of the scored boxes).
Only pixels with data in both datasets count.

Writes results/v7/<name>/:
    by_box_and_range.csv   every number, one row per box x range (with the pixel count)
    metrics.json           the overall numbers, in the same form as v6 results
    pred_<box>.tif         predicted biomass for the whole box on the CCI grid
                           (band 1 Mg/ha, band 2: 0 = train block, 1 = val, 2 = test)
    scatter.png            predicted against CCI, test pixels
    run_evaluate_*.json    the exact command and settings
Stops if results/v7/<name>/ already exists: nothing is overwritten.
"""
import os, re, csv, json, argparse
from datetime import datetime

import numpy as np
import torch
import rasterio
from rasterio.transform import Affine

from src import evaluate as V6                      # the v6 evaluation code: metrics, model loading, ridge
from src.dataset.dataset import BiomassDataset
from src.v7.common import BOXES, BOX_PIX, BLOCK_PIX, PIX_PER_DEG, DATA_ROOT, snap, write_run_record

RANGES = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 400), (400, None)]
ROLE = {"train": 0, "val": 1, "test": 2}
SEED, RATIO = 42, (0.7, 0.15, 1.0 - 0.7 - 0.15)
NAME_RE = re.compile(r"^(.*)_r(\d+)_c(\d+)$")


def box_dir(box):
    return os.path.join(DATA_ROOT, "blocks", box)


def range_label(lo, hi):
    return f"{lo}-{hi}" if hi else f">{lo}"


def numbers(pred, true, sd):
    m = V6.metrics(pred, true)
    m.update(V6.sd_metrics(pred, true, sd))
    return m


def predict_box(box, predict, norm_stats):
    """Predict all 25 blocks of a box. Returns box-sized arrays: pred, true, sd, role, ok."""
    pred = np.full((BOX_PIX, BOX_PIX), np.nan, np.float32)
    true = np.zeros((BOX_PIX, BOX_PIX), np.float32); sd = np.zeros_like(true)
    role = np.full((BOX_PIX, BOX_PIX), 255, np.uint8); ok = np.zeros((BOX_PIX, BOX_PIX), bool)
    for split in ("train", "val", "test"):
        ds = BiomassDataset(box_dir(box), split=split, split_ratio=RATIO, use_ae=True, seed=SEED, norm_stats=norm_stats)
        for i in range(len(ds)):
            x, y, valid, name = ds[i]
            r, c = (int(v) for v in NAME_RE.match(name).groups()[1:])
            sl = (slice(r * BLOCK_PIX, (r + 1) * BLOCK_PIX), slice(c * BLOCK_PIX, (c + 1) * BLOCK_PIX))
            y2 = y.squeeze(-1)
            good = (valid.bool() & V6.target_mask(y2)).numpy()
            p = predict(x, box)
            pred[sl] = np.where(good, p, np.nan)
            true[sl] = y2.numpy(); sd[sl] = np.load(ds.sd_paths[name]).squeeze()
            role[sl] = ROLE[split]; ok[sl] = good
    assert (role != 255).all(), f"{box}: not all 25 blocks found"
    return pred, true, sd, role, ok


def write_geotiff(path, box, pred, role):
    s = snap(box)["snapped"]
    pix = 1.0 / PIX_PER_DEG
    with rasterio.open(path, "w", driver="GTiff", height=BOX_PIX, width=BOX_PIX, count=2, dtype="float32", crs="EPSG:4326",
                       transform=Affine(pix, 0, s["west"], 0, -pix, s["north"]), compress="deflate", nodata=float("nan")) as dst:
        dst.write(pred, 1); dst.write(role.astype(np.float32), 2)
        dst.set_band_description(1, "predicted above-ground biomass, Mg/ha")
        dst.set_band_description(2, "block role: 0 train, 1 val, 2 test")


def run(name, model_name, checkpoint=None, train_boxes=None, score_boxes=None, pile="test", out_root="results/v7"):
    import types
    args = types.SimpleNamespace(name=name, model_name=model_name, checkpoint=checkpoint, pile=pile, out_root=out_root,
                                 train_boxes=",".join(train_boxes or BOXES), score_boxes=",".join(score_boxes or BOXES))
    out_dir = os.path.join(args.out_root, args.name)
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} already exists; nothing is overwritten. Choose another --name.")
    if pile not in ("test", "all"):
        raise SystemExit("--pile must be test or all")
    score_boxes = [b.strip() for b in args.score_boxes.split(",") if b.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats, train_info = None, {}

    # ---- the predictor: a function (x of one block, box name) -> (225, 225) array in Mg/ha ----
    if args.model_name in ("SmallCNN", "PointWiseModel"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for a trained model")
        meta = V6.load_train_meta(args.checkpoint)
        train_boxes = [os.path.basename(d.strip().rstrip("/")) for d in meta["data_dir"].split(",")]
        norm_stats = meta["norm_stats"] if meta.get("normalize") else None
        scale = float(meta.get("target_scale", 1.0))
        model = V6.MODELS[args.model_name](64).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        train_info = {"checkpoint": args.checkpoint, **{k: v for k, v in meta.items() if k != "norm_stats"}}

        def predict(x, box):
            with torch.no_grad():
                return (model(x[None].to(device)).cpu()[0] * scale).numpy()
    else:
        train_boxes = [b.strip() for b in args.train_boxes.split(",") if b.strip()]
        train_dirs = ",".join(box_dir(b) for b in train_boxes)
        if args.model_name == "Ridge":
            ridge, n_tiles = V6.fit_ridge(V6.Config(model_name="Ridge", data_dir=train_dirs))

            def predict(x, box):
                h, w, c = x.shape
                return ridge.predict(x.numpy().reshape(-1, c)).reshape(h, w).astype(np.float32)
        else:
            sums, counts = {}, {}
            for b in train_boxes:
                ds = BiomassDataset(box_dir(b), split="train", split_ratio=RATIO, use_ae=True, seed=SEED)
                n_tiles = len(ds)
                for i in range(len(ds)):
                    _, y, valid, _ = ds[i]
                    y2 = y.squeeze(-1); good = valid.bool() & V6.target_mask(y2)
                    sums[b] = sums.get(b, 0.0) + float(y2[good].sum()); counts[b] = counts.get(b, 0) + int(good.sum())
            overall_mean = sum(sums.values()) / sum(counts.values())
            box_means = {b: sums[b] / counts[b] for b in train_boxes}
            print(f"mean biomass of the train blocks: all {overall_mean:.1f} Mg/ha; per box {json.dumps({b: round(v, 1) for b, v in box_means.items()})}")

            def predict(x, box):
                value = box_means.get(box, overall_mean) if args.model_name == "BoxMean" else overall_mean
                return np.full(x.shape[:2], value, np.float32)
        train_info = {"checkpoint": None, "model_name": args.model_name, "data_dir": train_dirs, "seed": SEED,
                      "n_train_tiles": 17 * len(train_boxes)}

    seen = [b for b in score_boxes if b in train_boxes]
    if args.pile == "all" and seen:
        raise SystemExit(f"--pile all is only fair for boxes the model never trained on; it trained on {', '.join(seen)}")

    # ---- predict every scored box ----
    os.makedirs(out_dir)
    rows, pooled = [], {args.pile: []}
    for box in score_boxes:
        pred, true, sd, role, ok = predict_box(box, predict, norm_stats)
        write_geotiff(os.path.join(out_dir, f"pred_{box}.tif"), box, pred, role)
        trained_on_box = box in train_boxes
        piles = {args.pile: ok & (role == ROLE["test"]) if args.pile == "test" else ok}
        for pile, sel in piles.items():
            pooled[pile].append((box, pred[sel], true[sel], sd[sel]))
            for lo, hi in [(None, None)] + RANGES:
                rs = np.ones_like(true[sel], bool) if lo is None else ((true[sel] >= lo) & (true[sel] < hi) if hi else (true[sel] >= lo))
                if rs.sum() == 0:
                    rows.append({"box": box, "pile": pile, "range_mg_ha": "all" if lo is None else range_label(lo, hi), "n_pixels": 0})
                    continue
                rows.append({"box": box, "pile": pile, "range_mg_ha": "all" if lo is None else range_label(lo, hi),
                             **numbers(pred[sel][rs], true[sel][rs], sd[sel][rs])})
        left_out = int((~ok).sum())
        box_all = next(r for r in rows if r["box"] == box and r["range_mg_ha"] == "all")
        print(f"{box}: {args.pile} RMSE {box_all['rmse']:.1f} Mg/ha, R2 {box_all['r2']:.3f}"
              f"{'' if trained_on_box else '  (box never trained on)'}; {left_out} pixels left out")

    overall = {}
    for pile, parts in pooled.items():
        if not parts:
            continue
        P = np.concatenate([x[1] for x in parts]); T_ = np.concatenate([x[2] for x in parts]); S = np.concatenate([x[3] for x in parts])
        for lo, hi in [(None, None)] + RANGES:
            rs = np.ones_like(T_, bool) if lo is None else ((T_ >= lo) & (T_ < hi) if hi else (T_ >= lo))
            label = "all" if lo is None else range_label(lo, hi)
            if rs.sum() == 0:
                rows.append({"box": "all_scored_boxes", "pile": pile, "range_mg_ha": label, "n_pixels": 0}); continue
            m = numbers(P[rs], T_[rs], S[rs])
            rows.append({"box": "all_scored_boxes", "pile": pile, "range_mg_ha": label, **m})
            if lo is None:
                overall[pile] = m
        V6.scatter_plot(P, T_, overall[pile], f"{args.model_name} · {args.name}", os.path.join(out_dir, "scatter.png"))

    # ---- write ----
    cols = ["model", "name", "trained_on", "box", "pile", "range_mg_ha", "n_pixels", "rmse", "bias", "r2", "mae", "rmse_over_cci_sd",
            "frac_within_1sd", "true_mean", "pred_mean", "true_std", "cci_sd_rms"]
    trained_on = "+".join(train_boxes)
    with open(os.path.join(out_dir, "by_box_and_range.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"model": args.model_name, "name": args.name, "trained_on": trained_on,
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    summary = {
        "name": args.name, "evaluated": datetime.now().isoformat(timespec="seconds"),
        "config": {"model_name": args.model_name, "checkpoint": args.checkpoint, "data_dir": ",".join(box_dir(b) for b in score_boxes),
                   "split": args.pile, "seed": SEED, "split_train": 0.7, "split_val": 0.15, "out_root": args.out_root},
        "train": [train_info], "trained_on_boxes": train_boxes, "scored_on_boxes": score_boxes,
        "metrics": overall[args.pile],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_run_record(out_dir, "evaluate", settings={**vars(args), "trained_on_boxes": train_boxes, "device": str(device)}, results={"overall": overall[args.pile]})

    m = overall[args.pile]
    print(f"\n{args.name}: {args.pile} blocks of {len(score_boxes)} box(es): RMSE {m['rmse']:.1f}  bias {m['bias']:+.1f}  R2 {m['r2']:.3f}  "
          f"RMSE/CCI SD {m['rmse_over_cci_sd']:.2f}  n = {m['n_pixels']:,}")
    print(f"written to {out_dir}/")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="results/v7/<name>/")
    p.add_argument("--model_name", required=True, choices=["SmallCNN", "PointWiseModel", "Ridge", "ConstantMean", "BoxMean"])
    p.add_argument("--checkpoint", default=None, help="model.pth, for SmallCNN / PointWiseModel")
    p.add_argument("--train_boxes", default=",".join(BOXES), help="for the baselines: boxes whose train blocks are used")
    p.add_argument("--score_boxes", default=",".join(BOXES), help="boxes to score on")
    p.add_argument("--pile", default="test", choices=["test", "all"], help="test blocks, or all blocks of boxes never trained on")
    p.add_argument("--out_root", default="results/v7")
    a = p.parse_args()
    split = lambda v: [b.strip() for b in v.split(",") if b.strip()]
    run(a.name, a.model_name, a.checkpoint, split(a.train_boxes), split(a.score_boxes), a.pile, a.out_root)


if __name__ == "__main__":
    main()
