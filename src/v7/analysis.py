"""Extra tables from files that already exist. No model is run and nothing is trained.

    python -m src.v7.analysis                      # writes results/v7/analysis/
    sbatch setup/jasmin/v7/cpu_job.sbatch src.v7.analysis

It reads the saved prediction maps (results/v7/<name>/pred_<box>.tif), the saved scores
(results/v7/<name>/by_box_and_range.csv) and the CCI crops, and writes four tables:

  ranges_train_vs_test.csv   which biomass values the model saw in training, next to the values
                             in the test blocks it was scored on (one row per setup and side)
  bias_by_range.csv          bias and RMSE per biomass range, three seeds side by side
                             (gathered from the saved scores, nothing recomputed)
  seed_average.csv           the three seeds averaged into one prediction, scored on the test blocks,
                             next to the three single scores
  cells_by_scale.csv         the same scores after averaging prediction and CCI over cells of
                             3 x 3, 9 x 9 and 25 x 25 pixels (9 pixels is about 1 km north-south),
                             with CCI's own SD at 100 m and from CCI's own 1 km map
  checks.json                the checks listed below, with their outcome

Words: a *setup* is what the model was trained on (all6, noboreal, nonl, ...); a *seed* is the
starting seed of a run (42 for the first run, 1 and 2 for the repeats); a *cell* is a square of
k x k CCI pixels. Cells never cross a block edge, because 3, 9 and 25 all divide 225.

Checks (the script stops if the first one fails):
  1. every single-seed score recomputed here from the saved maps equals the saved score;
  2. the shares per biomass range add up to 100 %;
  3. the RMSE of the averaged prediction is never above the mean of the three single RMSE values;
  4. CCI's 100 m map averaged onto the grid of CCI's own 1 km map comes close to that map
     (correlation above 0.95 in every box). If not, the SD from the 1 km map is a rough yardstick only.
Stops if the output folder already exists: nothing is overwritten.
"""
import os, csv, json, argparse
from datetime import date

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, Resampling

from src.v7.common import BOXES, BOX_PIX, BLOCK_PIX, PIX_PER_DEG, crop_path, snap, pixel_size_m, write_run_record
from src.v7.evaluate import RANGES, ROLE, range_label, numbers
from src.v7.experiments import setups, BOX_CODE, MODELS

SEEDS = [("", 42), ("_s1", 1), ("_s2", 2)]          # (suffix in the folder name, seed)
CELLS = [1, 3, 9, 25]                               # cell side in pixels; 1 = single pixels
MIN_VALID = 0.9                                     # a cell counts if at least 90 % of its pixels have data
ONE_KM = "/neodc/esacci/biomass/data/agb/maps/v7.0/netcdf/ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc"
PCTS = [1, 5, 25, 50, 75, 95, 99]
LABELS = [range_label(lo, hi) for lo, hi in RANGES]


def in_range(v, lo, hi):
    return (v >= lo) & (v < hi) if hi else (v >= lo)


def load_box(box, results_root):
    """CCI biomass, CCI SD, the role of every pixel's block (0 train, 1 val, 2 test) and which pixels have data."""
    with rasterio.open(crop_path(box, "AGB")) as s:
        true = s.read(1).astype(np.float32)
    with rasterio.open(crop_path(box, "AGB_SD")) as s:
        sd = s.read(1).astype(np.float32)
    with rasterio.open(os.path.join(results_root, "all6_pw_on_all6_test", f"pred_{box}.tif")) as s:
        ok = np.isfinite(s.read(1)); role = s.read(2).astype(np.uint8)
    assert true.shape == (BOX_PIX, BOX_PIX) and role.shape == true.shape
    return {"true": true, "sd": sd, "role": role, "ok": ok}


def load_pred(results_root, setup, lab, code, box):
    with rasterio.open(os.path.join(results_root, f"{setup}_{lab}_on_{code}_test", f"pred_{box}.tif")) as s:
        return s.read(1)


def saved_rows(results_root, setup, lab, code):
    with open(os.path.join(results_root, f"{setup}_{lab}_on_{code}_test", "by_box_and_range.csv")) as f:
        return [r for r in csv.DictReader(f) if r["pile"] == "test"]


def r4(v):
    return round(float(v), 4) if v is not None and np.isfinite(v) else ""


# ───────────────────────────── table 1a: range in training against range in test ─────────────────────────────
def describe(v):
    out = {"n_pixels": int(v.size), "mean": r4(v.mean()), "spread_sd": r4(v.std())}
    for p, q in zip(PCTS, np.percentile(v, PCTS)):
        out[f"p{p}"] = r4(q)
    out["max"] = r4(v.max())
    for (lo, hi), lab in zip(RANGES, LABELS):
        out[f"pct_{lab}"] = r4(100.0 * in_range(v, lo, hi).mean())
    return out


def table_ranges(data, checks):
    rows, train_share = [], {}
    for name, setup in setups().items():
        code, boxes = next(iter(setup["score"].items()))
        tr = np.concatenate([data[b]["true"][(data[b]["role"] == ROLE["train"]) & data[b]["ok"]] for b in setup["train"]])
        d = describe(tr)
        train_share[name] = {lab: d[f"pct_{lab}"] for lab in LABELS}
        p99, top = np.percentile(tr, 99), tr.max()
        rows.append({"experiment": setup["experiment"], "setup": name, "side": "train blocks", "boxes": "+".join(setup["train"]), **d})
        parts = {b: data[b]["true"][(data[b]["role"] == ROLE["test"]) & data[b]["ok"]] for b in boxes}
        if len(boxes) > 1:
            parts["all_scored_boxes"] = np.concatenate(list(parts.values()))
        for b, te in parts.items():
            rows.append({"experiment": setup["experiment"], "setup": name, "side": "test blocks", "boxes": b, **describe(te),
                         "pct_above_train_p99": r4(100.0 * (te > p99).mean()), "pct_above_train_max": r4(100.0 * (te > top).mean()),
                         "trained_on_this_box": "yes" if b in setup["train"] or name == "all6" else "no"})
    sums = [sum(r[f"pct_{lab}"] for lab in LABELS) for r in rows]
    checks["2_range_shares_add_up"] = {"rows": len(rows), "lowest_sum": round(min(sums), 3), "highest_sum": round(max(sums), 3),
                                       "passed": bool(min(sums) > 99.99 and max(sums) < 100.01)}
    return rows, train_share


# ───────────────────────────── table 1b: bias and RMSE per range, from the saved scores ─────────────────────────────
def table_bias(results_root, train_share):
    rows = []
    for name, setup in setups().items():
        code, boxes = next(iter(setup["score"].items()))
        for m in MODELS:
            per_seed = [{(r["box"], r["range_mg_ha"]): r for r in saved_rows(results_root, name, m + suf, code)} for suf, _ in SEEDS]
            n_all = {b: int(per_seed[0][(b, "all")]["n_pixels"]) for b, rg in per_seed[0] if rg == "all"}
            for (box, rg), r0 in per_seed[0].items():
                n = int(r0["n_pixels"])
                row = {"experiment": setup["experiment"], "setup": name, "model": m, "box": box, "range_mg_ha": rg, "n_pixels": n,
                       "pct_of_test_pixels": r4(100.0 * n / n_all[box]),
                       "pct_of_train_pixels_in_this_range": "" if rg == "all" else train_share[name][rg]}
                if n:
                    for key in ("bias", "rmse"):
                        vals = [float(ps[(box, rg)][key]) for ps in per_seed]
                        for (_, seed), v in zip(SEEDS, vals):
                            row[f"{key}_seed{seed}"] = r4(v)
                        row[f"{key}_mean"] = r4(np.mean(vals)); row[f"{key}_lowest"] = r4(min(vals)); row[f"{key}_highest"] = r4(max(vals))
                    row["cci_mean"] = r4(float(r0["true_mean"]))
                    row["predicted_mean"] = r4(np.mean([float(ps[(box, rg)]["pred_mean"]) for ps in per_seed]))
                rows.append(row)
    return rows


# ───────────────────────────── tables 3 and 2: the seed average, and averaged cells ─────────────────────────────
def to_cells(a, k):
    n = BOX_PIX // k
    return a.reshape(n, k, n, k).sum(axis=(1, 3))


def cells_of(pred, d, k):
    """Prediction, CCI and CCI SD averaged over k x k cells of the test blocks. k = 1 gives the single pixels."""
    ok = d["ok"] & np.isfinite(pred)
    n = to_cells(ok.astype(np.float64), k)
    keep = (d["role"][::k, ::k] == ROLE["test"]) & (n >= MIN_VALID * k * k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = lambda a: (to_cells(np.where(ok, a, 0).astype(np.float64), k) / n)[keep]
        p, t, s_shared = mean(pred), mean(d["true"]), mean(d["sd"])
        s_indep = (np.sqrt(to_cells(np.where(ok, d["sd"].astype(np.float64) ** 2, 0), k)) / n)[keep]
    return p.astype(np.float32), t.astype(np.float32), s_shared.astype(np.float32), s_indep.astype(np.float32)


def tables_seeds_and_cells(data, results_root, one_km, checks):
    seed_rows, cell_rows = [], []
    compared, worst, bad = 0, 0.0, []
    for name, setup in setups().items():
        code, boxes = next(iter(setup["score"].items()))
        for m in MODELS:
            saved = [{r["box"]: r for r in saved_rows(results_root, name, m + suf, code) if r["range_mg_ha"] == "all"} for suf, _ in SEEDS]
            preds = {b: [load_pred(results_root, name, m + suf, code, b) for suf, _ in SEEDS] for b in boxes}
            avg = {b: np.mean(np.stack(preds[b]), axis=0, dtype=np.float32) for b in boxes}
            groups = [(b, [b]) for b in boxes] + ([("all_scored_boxes", boxes)] if len(boxes) > 1 else [])
            for k in CELLS:
                per_box = {b: {"avg": cells_of(avg[b], data[b], k), "seeds": [cells_of(p, data[b], k) for p in preds[b]]} for b in boxes}
                for label, members in groups:
                    cat = lambda pick: [np.concatenate([pick(per_box[b])[i] for b in members]) for i in range(4)]
                    pa, ta, sa_shared, sa_indep = cat(lambda x: x["avg"])
                    single = []
                    for si in range(len(SEEDS)):
                        ps, ts, ss, _ = cat(lambda x: x["seeds"][si])
                        single.append(numbers(ps, ts, ss))
                    ma = numbers(pa, ta, sa_shared)

                    if k == 1:
                        # check 1: the single-seed scores recomputed here equal the saved ones
                        for (suf, seed), s_new, s_old in zip(SEEDS, single, saved):
                            old = s_old[label]
                            diffs = [abs(s_new[key] - float(old[key])) for key in ("rmse", "bias", "r2")]
                            compared += 1; worst = max(worst, *diffs)
                            if max(diffs) > 2e-3 or s_new["n_pixels"] != int(old["n_pixels"]):
                                bad.append(f"{name} {m}{suf} {label}: here {s_new['rmse']:.4f}/{s_new['bias']:.4f}/{s_new['r2']:.4f}/{s_new['n_pixels']}, "
                                           f"saved {old['rmse']}/{old['bias']}/{old['r2']}/{old['n_pixels']}")
                        row = {"experiment": setup["experiment"], "setup": name, "model": m, "box": label, "n_pixels": ma["n_pixels"]}
                        for key in ("rmse", "bias", "r2"):
                            vals = [s[key] for s in single]
                            for (_, seed), v in zip(SEEDS, vals):
                                row[f"{key}_seed{seed}"] = r4(v)
                            row[f"{key}_mean_of_seeds"] = r4(np.mean(vals))
                            row[f"{key}_seed_average"] = r4(ma[key])
                        rm = [s["rmse"] for s in single]
                        row["rmse_gain_over_mean_of_seeds"] = r4(np.mean(rm) - ma["rmse"])
                        row["rmse_best_single_seed"] = r4(min(rm))
                        row["seed_average_beats_best_seed"] = "yes" if ma["rmse"] < min(rm) else "no"
                        row["rmse_over_cci_sd_seed_average"] = r4(ma["rmse_over_cci_sd"])
                        seed_rows.append(row)

                    lat = np.mean([(snap(b)["snapped"]["north"] + snap(b)["snapped"]["south"]) / 2 for b in members])
                    ns, ew = pixel_size_m(lat)
                    crow = {"experiment": setup["experiment"], "setup": name, "model": m, "box": label, "cell_pixels": k,
                            "cell_km_north_south": round(k * ns / 1000, 2), "cell_km_east_west": round(k * ew / 1000, 2) if len(members) == 1 else "",
                            "n_cells": ma["n_pixels"],
                            "rmse_seed_average": r4(ma["rmse"]), "bias_seed_average": r4(ma["bias"]), "r2_seed_average": r4(ma["r2"]),
                            "rmse_mean_of_seeds": r4(np.mean([s["rmse"] for s in single])),
                            "r2_mean_of_seeds": r4(np.mean([s["r2"] for s in single])),
                            "cci_spread_between_cells": r4(ma["true_std"]),
                            "cci_sd_if_errors_fully_shared": r4(np.sqrt(np.mean(sa_shared.astype(np.float64) ** 2))),
                            "cci_sd_if_errors_independent": r4(np.sqrt(np.mean(sa_indep.astype(np.float64) ** 2)))}
                    if k == 9 and one_km and all(b in one_km for b in members):
                        sq = np.concatenate([one_km[b]["sd_test_cells"] for b in members]).astype(np.float64)
                        crow["cci_sd_from_cci_1km_map"] = r4(np.sqrt(np.mean(sq ** 2)))
                        crow["rmse_seed_average_over_cci_1km_sd"] = r4(ma["rmse"] / np.sqrt(np.mean(sq ** 2)))
                    cell_rows.append(crow)

    checks["1_recomputed_scores_equal_saved"] = {"scores_compared": compared, "largest_difference": round(worst, 5),
                                                 "passed": not bad, "problems": bad}
    if bad:
        raise SystemExit("check 1 failed: scores recomputed from the saved maps differ from the saved scores:\n  " + "\n  ".join(bad[:10]))
    over = [r for r in seed_rows if float(r["rmse_seed_average"]) > float(r["rmse_mean_of_seeds"]) + 1e-3]
    checks["3_seed_average_not_worse_than_mean_of_seeds"] = {"rows": len(seed_rows), "rows_where_it_is_worse": len(over), "passed": not over}
    return seed_rows, cell_rows


# ───────────────────────────── CCI's own 1 km map (for its SD at 1 km, and check 4) ─────────────────────────────
def read_one_km(data, checks):
    """For each box: CCI's own SD at 1 km on the cells whose centre lies in a test block. Also check 4:
    our 100 m crop averaged onto the 1 km grid, against CCI's own 1 km biomass."""
    out, report = {}, {}
    target = (date(2020, 1, 1) - date(1990, 1, 1)).days
    with rasterio.open(f'NETCDF:"{ONE_KM}":agb') as a, rasterio.open(f'NETCDF:"{ONE_KM}":agb_sd') as b:
        assert a.transform.e < 0, "expected a north-up grid"
        band = next(i for i in range(1, a.count + 1) if int(float(a.tags(i)["NETCDF_DIM_time"])) == target)
        for box in BOXES:
            s = snap(box)["snapped"]
            win = from_bounds(s["west"], s["south"], s["east"], s["north"], a.transform).round_offsets().round_lengths()
            agb = a.read(band, window=win).astype(np.float32); sd = b.read(band, window=win).astype(np.float32)
            tr = a.window_transform(win)
            good = (agb >= 0) & (sd >= 0)
            # centre of every 1 km cell -> the 100 m pixel it falls on -> the role of that pixel's block
            rows_i, cols_j = np.indices(agb.shape)
            lon = tr.c + (cols_j + 0.5) * tr.a; lat = tr.f + (rows_i + 0.5) * tr.e
            col = np.floor((lon - s["west"]) * PIX_PER_DEG).astype(int); row = np.floor((s["north"] - lat) * PIX_PER_DEG).astype(int)
            inside = (col >= 0) & (col < BOX_PIX) & (row >= 0) & (row < BOX_PIX)
            role = np.full(agb.shape, 255, np.uint8)
            role[inside] = data[box]["role"][row[inside], col[inside]]
            test = good & (role == ROLE["test"])
            out[box] = {"sd_test_cells": sd[test]}
            # check 4: the 100 m crop averaged onto this 1 km grid
            pix = 1.0 / PIX_PER_DEG
            mine = np.full(agb.shape, np.nan, np.float32)
            reproject(data[box]["true"], mine, src_transform=Affine(pix, 0, s["west"], 0, -pix, s["north"]), src_crs="EPSG:4326",
                      dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.average, dst_nodata=np.nan)
            inner = good & np.isfinite(mine)
            inner[[0, -1], :] = False; inner[:, [0, -1]] = False          # edge cells are only partly inside the box
            diff = mine[inner] - agb[inner]
            corr = float(np.corrcoef(mine[inner], agb[inner])[0, 1])
            # is the 1 km grid read at the right place? moving it by one cell must make the match worse
            shifted = {}
            for where, (dy, dx) in {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}.items():
                other = a.read(band, window=Window(win.col_off + dx, win.row_off + dy, win.width, win.height)).astype(np.float32)
                both = inner & (other >= 0)
                shifted[where] = round(float(np.corrcoef(mine[both], other[both])[0, 1]), 4)
            report[box] = {"cells_compared": int(inner.sum()), "cells_in_test_blocks": int(test.sum()),
                           "correlation": round(corr, 4), "correlation_if_moved_one_cell": shifted, "no_move_is_best": bool(corr > max(shifted.values())),
                           "mean_difference_mg_ha": round(float(diff.mean()), 2), "rmse_mg_ha": round(float(np.sqrt((diff ** 2).mean())), 2),
                           "spread_between_cells_100m_average": round(float(mine[inner].std()), 1),
                           "spread_between_cells_cci_1km_map": round(float(agb[inner].std()), 1),
                           "cci_1km_mean": round(float(agb[inner].mean()), 1), "cci_1km_sd_rms_test_blocks": round(float(np.sqrt((sd[test].astype(np.float64) ** 2).mean())), 1)}
    checks["4_100m_average_against_cci_1km_map"] = {
        "file": ONE_KM, "year_band": band, "per_box": report,
        "passed": bool(min(r["correlation"] for r in report.values()) > 0.95),
        "if_not_passed": "CCI's own 1 km map is then not simply its 100 m map averaged onto the 1 km grid. "
                         "The SD taken from the 1 km map is a rough yardstick only, not an exact partner of the 9 x 9 cells."}
    return out


def write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows):5d} rows -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default="results/v7", help="where the saved results folders are")
    p.add_argument("--out_dir", default="results/v7/analysis")
    a = p.parse_args()
    if os.path.exists(a.out_dir):
        raise SystemExit(f"{a.out_dir} already exists; nothing is overwritten. Choose another --out_dir.")

    checks = {}
    data = {b: load_box(b, a.results_root) for b in BOXES}
    range_rows, train_share = table_ranges(data, checks)
    bias_rows = table_bias(a.results_root, train_share)
    try:
        one_km = read_one_km(data, checks)
    except Exception as e:                                   # the other three tables do not depend on it
        one_km = None
        checks["4_100m_average_against_cci_1km_map"] = {"file": ONE_KM, "passed": False, "error": repr(e)}
        print(f"CCI's 1 km map could not be read ({e!r}); the 1 km SD columns stay empty")
    seed_rows, cell_rows = tables_seeds_and_cells(data, a.results_root, one_km, checks)

    os.makedirs(a.out_dir)
    write_csv(os.path.join(a.out_dir, "ranges_train_vs_test.csv"), range_rows)
    write_csv(os.path.join(a.out_dir, "bias_by_range.csv"), bias_rows)
    write_csv(os.path.join(a.out_dir, "seed_average.csv"), seed_rows)
    write_csv(os.path.join(a.out_dir, "cells_by_scale.csv"), cell_rows)
    with open(os.path.join(a.out_dir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    for name, c in checks.items():
        print(f"check {name}: {'passed' if c['passed'] else 'NOT PASSED'}")
    write_run_record(a.out_dir, "analysis", settings={**vars(a), "seeds": [s for _, s in SEEDS], "cell_pixels": CELLS, "min_valid_share": MIN_VALID,
                                                      "ranges_mg_ha": LABELS, "one_km_file": ONE_KM},
                     results={"checks_passed": {k: v["passed"] for k, v in checks.items()}})


if __name__ == "__main__":
    main()
