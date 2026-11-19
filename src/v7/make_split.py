"""Step 5 of the v7 setup: the train / val / test split, by whole blocks, saved to disk.

    python -m src.v7.make_split

Method (spatial block cross-validation): each box is cut into 25 blocks; a whole
block is train, val or test. Inside each box the 25 block names are sorted, shuffled
with seed 42 and dealt 70 % / 15 % / 15 %, which gives 17 train, 3 val and 5 test
blocks per box. Every box therefore contributes to all three.

This uses the very function train.py and evaluate.py already use for a data folder
(BiomassDataset._default_split), so the saved split IS the split the models get,
for both models, and v6 code is not changed. src/v7/train.py checks it again at the start
of every training run.

Writes <DATA_ROOT>/split_v7_seed42.json and a copy plus a readable summary in results/v7/split/.
Stops if the split file already exists with different content: nothing is overwritten.
"""
import os, csv, json

import numpy as np
import rasterio

from src.dataset.dataset import BiomassDataset
from src.v7.common import BOXES, BLOCK_PIX, BLOCKS_PER_SIDE, DATA_ROOT, crop_path, blocks, write_run_record

SEED = 42
RATIO = (0.7, 0.15, 1.0 - 0.7 - 0.15)      # exactly what train.py passes: (split_train, split_val, the rest)
LETTER = {"train": ".", "val": "v", "test": "T"}


def split_for(box):
    names = [f"{box}_{name}" for name, _, _ in blocks()]
    return {s: BiomassDataset._default_split(names, s, split_ratio=RATIO, seed=SEED) for s in ("train", "val", "test")}


def main():
    split = {"seed": SEED, "ratio_train_val_test": [0.7, 0.15, 0.15], "unit": "block (225 x 225 CCI pixels)",
             "rule": "per box: sorted block names, shuffled with numpy default_rng(seed), first 70 % train, next 15 % val, rest test",
             "boxes": {box: split_for(box) for box in BOXES}}

    rows, maps = [], []
    for box in BOXES:
        role = {n: s for s in ("train", "val", "test") for n in split["boxes"][box][s]}
        assert len(role) == BLOCKS_PER_SIDE ** 2, f"{box}: a block is missing from, or twice in, the split"
        with rasterio.open(crop_path(box, "AGB")) as src:
            agb = src.read(1)
        maps.append(f"{box}  (north at the top; . = train, v = val, T = test)")
        for r in range(BLOCKS_PER_SIDE):
            maps.append("    " + " ".join(LETTER[role[f"{box}_r{r}_c{c}"]] for c in range(BLOCKS_PER_SIDE)))
        for s in ("train", "val", "test"):
            px = np.concatenate([agb[r * BLOCK_PIX:(r + 1) * BLOCK_PIX, c * BLOCK_PIX:(c + 1) * BLOCK_PIX].ravel()
                                 for name, r, c in blocks() if role[f"{box}_{name}"] == s])
            rows.append({"box": box, "split": s, "blocks": len(split["boxes"][box][s]), "pixels": int(px.size),
                         "agb_mean": round(float(px.mean()), 1), "agb_pct_zero": round(100 * float((px == 0).mean()), 1),
                         "agb_pct_150_up": round(100 * float((px >= 150).mean()), 1)})

    path = os.path.join(DATA_ROOT, "split_v7_seed42.json")
    if os.path.exists(path):
        with open(path) as f:
            if json.load(f) != split:
                raise SystemExit(f"{path} exists with a different split; not overwritten")
        print(f"split file already there and identical: {path}")
    else:
        with open(path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"split -> {path}")

    out = "results/v7/split"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "split_v7_seed42.json"), "w") as f:
        json.dump(split, f, indent=2)
    with open(os.path.join(out, "split_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "split_maps.txt"), "w") as f:
        f.write("\n".join(maps) + "\n")

    print("\n".join(maps))
    print(f"\n{'box':18s}{'split':6s}{'blocks':>7s}{'pixels':>10s}{'mean':>8s}{'zero %':>8s}{'150+ %':>8s}")
    for r in rows:
        print(f"{r['box']:18s}{r['split']:6s}{r['blocks']:>7d}{r['pixels']:>10,d}{r['agb_mean']:>8.1f}{r['agb_pct_zero']:>8.1f}{r['agb_pct_150_up']:>8.1f}")
    tot = {s: sum(r["blocks"] for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"\nall boxes: {tot['train']} train, {tot['val']} val, {tot['test']} test blocks")
    write_run_record(out, "make_split", settings={"seed": SEED, "ratio": RATIO, "block_pixels": BLOCK_PIX}, results={"blocks": tot, "summary": rows})


if __name__ == "__main__":
    main()
