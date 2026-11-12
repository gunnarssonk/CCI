"""Collect every evaluation into one table.

Reads results/*/metrics.json (written by src/evaluate.py) and writes
results/all_results.csv, one row per evaluation. The file is rebuilt from
scratch on every run, so it can never go stale: just run it again.

    python -m src.collect_results

Where the training columns (trained_on, n_train_tiles, init_seed) come from,
in this order:
  1. the "train" block inside metrics.json (evaluate.py saves it since 2026-11-12);
  2. the train_meta.json next to the checkpoint, if that checkpoint folder
     exists on this machine;
  3. otherwise the cell stays empty. Nothing is guessed.
"""
import os, json, csv, argparse

COLUMNS = [
    "name", "model", "trained_on", "n_train_tiles", "init_seed",
    "scored_on", "split", "rmse", "mae", "r2", "bias",
    "rmse_over_cci_sd", "frac_within_1sd", "date", "checkpoint_path",
]
METRIC_COLUMNS = ["rmse", "mae", "r2", "bias", "rmse_over_cci_sd", "frac_within_1sd"]
MEMBER_SEP = ";"   # between the members of an ensemble, e.g. init_seed "0;1;2"


def folder_names(data_dir):
    """'/long/path/data_a,/long/path/data_b' -> 'data_a+data_b' (same label as
    data_tag in dataset.py, re-done here so this script needs no torch import)."""
    if not data_dir:
        return ""
    parts = [d.strip().rstrip("/") for d in str(data_dir).split(",") if d.strip()]
    return "+".join(os.path.basename(p) for p in parts)


def cell(v):
    """One CSV cell: empty for 'not recorded', numbers rounded to 4 decimals."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def join_members(values):
    """One value per ensemble member -> one cell. All the same -> written once;
    otherwise joined with ';' (an unknown member leaves a gap, e.g. '0;;2')."""
    cells = [cell(v) for v in values]
    if len(set(cells)) <= 1:
        return cells[0] if cells else ""
    return MEMBER_SEP.join(cells)


def train_info_from_disk(checkpoint):
    """train_meta.json next to model.pth, or None if it is not on this machine."""
    path = os.path.join(os.path.dirname(checkpoint), "train_meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        meta = json.load(f)
    meta.pop("norm_stats", None)
    return meta


def effective_init_seed(info):
    """The seed that set the starting weights. train.py uses `seed` when
    --init_seed was not given (recorded as null). If the key is missing
    altogether (runs before 2026-09-24) it was not recorded: leave empty."""
    if "init_seed" not in info:
        return None
    return info["init_seed"] if info["init_seed"] is not None else info.get("seed")


def build_row(folder, summary):
    cfg = summary.get("config", {})
    m = summary["metrics"]
    model_name = cfg.get("model_name")
    is_ridge = model_name == "Ridge"
    ckpts = [c.strip() for c in (cfg.get("checkpoint") or "").split(",") if c.strip()]

    # ---- training info: one dict per checkpoint (or None when unknown) ----
    if "train" in summary:
        infos = summary["train"]
    elif is_ridge:
        # Ridge is fitted inside evaluate.py on the train split of the scored folder.
        infos = [{"data_dir": cfg.get("data_dir")}]
    else:
        infos = [train_info_from_disk(c) for c in ckpts]
    missing_train_info = (not infos) or any(i is None or not i.get("data_dir") for i in infos)
    infos = [i or {} for i in infos]

    model = model_name or ""
    if len(ckpts) > 1:
        model = f"{model} x{len(ckpts)}"

    row = {
        "name": folder,
        "model": model,
        "trained_on": join_members([folder_names(i.get("data_dir")) or None for i in infos]),
        "n_train_tiles": join_members([i.get("n_train_tiles") for i in infos]),
        "init_seed": "" if is_ridge else join_members([effective_init_seed(i) for i in infos]),
        "scored_on": folder_names(cfg.get("data_dir")),
        "split": cell(cfg.get("split")),
        "date": (summary.get("evaluated") or "")[:10],
        "checkpoint_path": MEMBER_SEP.join(ckpts),
    }
    for k in METRIC_COLUMNS:
        row[k] = cell(m.get(k))
    return row, summary.get("evaluated") or "", missing_train_info


def collect(results_dir="results", out_name="all_results.csv"):
    rows, skipped, no_train_info = [], [], []
    for folder in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, folder)
        if not os.path.isdir(path):
            continue   # loose files such as all_results.csv itself
        mpath = os.path.join(path, "metrics.json")
        if not os.path.exists(mpath):
            skipped.append((folder, "no metrics.json"))
            continue
        try:
            with open(mpath) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((folder, f"metrics.json unreadable ({e})"))
            continue
        if not isinstance(summary.get("metrics"), dict):
            skipped.append((folder, "metrics.json has no 'metrics' block"))
            continue
        row, stamp, missing = build_row(folder, summary)
        rows.append((stamp, row))
        if missing:
            no_train_info.append(folder)

    rows.sort(key=lambda r: (r[0], r[1]["name"]))   # oldest evaluation first
    out_path = os.path.join(results_dir, out_name)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(row for _, row in rows)

    print(f"{len(rows)} rows written to {out_path}")
    if no_train_info:
        print(f"{len(no_train_info)} rows have empty training columns "
              f"(no train_meta.json for the checkpoint on this machine):")
        for folder in no_train_info:
            print(f"  {folder}")
    if skipped:
        print(f"{len(skipped)} folders skipped:")
        for folder, why in skipped:
            print(f"  {folder}: {why}")
    else:
        print("No folders skipped.")
    return rows, skipped


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()
    collect(args.results_dir)
