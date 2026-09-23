"""Fetch CCI's per-pixel uncertainty (the SD band) for tiles that were downloaded
before build.py saved it. Writes <data_dir>/targets_sd/<name>_sd.npy for every
tile in <data_dir>/targets/, skipping ones that already exist.

    python -m src.dataset.fetch_sd --data_dir data_gee/data_france_2020_v2

Tile centre lat/lon is parsed from the filename; buffer, year and size come from
manifest.json. Note the SD comes from a separate request to the one that produced
the stored AGB, so ~3% of pixels may be shifted by one source pixel relative to
it (nearest-neighbour resampling jitter). Fine for aggregate comparisons.
"""
import os, re, json, argparse, time

import ee
import numpy as np
from tqdm import tqdm

from src.dataset.build import build_geom, fetch_agb

NAME_RE = re.compile(r"lat([+-]\d+\.\d+)_lon([+-]\d+\.\d+)_(\d+)")


def main(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        run = json.load(f)["runs"][-1]
    buffer_deg, year, dim = run["buffer_deg"], run["year"], run["master_dim"]

    tgt_dir = os.path.join(data_dir, "targets")
    sd_dir = os.path.join(data_dir, "targets_sd")
    os.makedirs(sd_dir, exist_ok=True)

    names = sorted(f[:-len("_y.npy")] for f in os.listdir(tgt_dir) if f.endswith("_y.npy"))
    todo = [n for n in names if not os.path.exists(os.path.join(sd_dir, f"{n}_sd.npy"))]
    print(f"{len(names)} tiles, {len(todo)} without SD yet")

    failed = []
    for name in tqdm(todo, unit="tile"):
        m = NAME_RE.match(name)
        lat, lon = float(m.group(1)), float(m.group(2))
        geom, _ = build_geom(lat, lon, buffer_deg)
        try:
            fetched = fetch_agb(geom, year, dim, with_sd=True)
            if fetched is None:
                failed.append(name); continue
            _, sd = fetched
            np.save(os.path.join(sd_dir, f"{name}_sd.npy"), sd)
            time.sleep(0.5)
        except Exception as e:
            print(f"  {name}: {e}")
            failed.append(name)

    print(f"done: {len(todo) - len(failed)} written, {len(failed)} failed")
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    ee.Initialize(project=os.getenv("EE_PROJECT", "esa-cci"))
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    main(p.parse_args().data_dir)
