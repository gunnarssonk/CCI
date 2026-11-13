"""Build a dataset folder with AlphaEarth embeddings from Google Earth Engine and
targets from the ESA Climate Toolbox (any CCI dataset), in the same layout that
build_gee.py produces, so train.py and evaluate.py work unchanged:

    <output_dir>/ae_embeddings/<name>_ae.npy   (H, W, 64)
    <output_dir>/targets/<name>_y.npy          (H, W, 1)
    <output_dir>/targets_sd/<name>_sd.npy      (H, W, 1)   if the dataset has an uncertainty variable
    <output_dir>/manifest.json                 provenance, incl. a "target" block

Runs in the `ect` environment (setup/default/ect_env.sh):

    mamba activate ect
    python -m src.dataset.build_ect --country France --year 2020 --total_samples 100 --output_dir data_gee/data_france_2020_ect
    python -m src.dataset.build_ect ... --data_id esacci.<other ECV dataset id>

The GEE-only builder (build_gee.py) is untouched; this reuses its point sampling,
geometry and embedding download by import.

New tiles never share ground with a tile that already exists (since 2026-11-13):
the ones made earlier in the same run, the ones already in --output_dir, and the
ones in any folder listed in --avoid_dirs. See run_build(). build_gee.py's own
loop does not have this check.

    python -m src.dataset.build_jasmin ... --output_dir $D/data_france_2020_jtest150 \
        --avoid_dirs $D/data_france_2020_j300,$D/data_france_2020_j700
    python -m src.dataset.build_jasmin ... --allow_overlap true     # the old behaviour

Used by:      build_jasmin.py borrows run_build() and args_for() from this file,
              so do not delete it even if the toolbox itself is never used.
Borrows from: build_gee.py (sampling, AlphaEarth download) and targets_ect.py.
"""
import os, json, time, argparse
from dataclasses import dataclass, fields

import ee
import numpy as np
from tqdm import tqdm

from src.dataset.build_gee import build_geom, fetch_alphaearth, sample_point_in_zone, write_manifest, str2bool, refuse_unless_append
from src.dataset.targets_ect import EctTarget, DEFAULT_DATA_ID, DEFAULT_STORE, NAME_RE

# Give up when this many random locations in a row all overlap an existing tile:
# by then less than about 1 % of the country is still free.
MAX_SKIPS_IN_A_ROW = 500


@dataclass
class Config:
    output_dir: str = None          # required
    master_dim: int = 256
    buffer_deg: float = 0.09
    total_samples: int = 100
    max_tries: int = 1000
    year: int = 2020
    country: str = "France"

    # target from the toolbox
    data_id: str = DEFAULT_DATA_ID
    store_id: str = DEFAULT_STORE
    var: str = None                 # value variable; auto-detected if None
    sd_var: str = None              # uncertainty variable; auto-detected if None
    method: str = "nearest"         # regridding: nearest | linear
    fill_value: float = 0.0         # written where the source has no data
    append: bool = False            # add tiles to a folder that already has some
    avoid_dirs: str = None          # other dataset folders (comma-separated) whose tiles the new ones must not overlap
    allow_overlap: bool = False     # true = old behaviour, tiles may share ground with each other

    def __post_init__(self):
        if not self.output_dir:
            raise SystemExit("--output_dir is required, e.g. --output_dir data_gee/data_france_2020_ect")
        self.ae_dir = os.path.join(self.output_dir, "ae_embeddings")
        self.target_dir = os.path.join(self.output_dir, "targets")
        self.sd_dir = os.path.join(self.output_dir, "targets_sd")
        refuse_unless_append(self.ae_dir, self.output_dir, self.append)
        for d in (self.output_dir, self.ae_dir, self.target_dir, self.sd_dir):
            os.makedirs(d, exist_ok=True)


def args_for(config_cls, parser=None):
    """Every field of a Config dataclass as a CLI flag (same pattern as build_gee.py)."""
    parser = parser or argparse.ArgumentParser()
    for f in fields(config_cls):
        t = str2bool if f.type is bool else f.type
        parser.add_argument(f"--{f.name}", type=t, default=f.default)
    return vars(parser.parse_args())


def args_extract(parser):
    return args_for(Config, parser)


def existing_tile_centres(folders):
    """(lat, lon) of the centre of every tile in these dataset folders, read from
    the file names in ae_embeddings/ (lat+45.8388_lon-000.1570_00100_ae.npy)."""
    centres = []
    for folder in folders:
        ae_dir = os.path.join(folder, "ae_embeddings")
        if not os.path.isdir(ae_dir):
            continue
        for f in os.listdir(ae_dir):
            m = NAME_RE.match(f)
            if m and f.endswith("_ae.npy"):
                centres.append((float(m.group(1)), float(m.group(2))))
    return centres


def overlaps_existing(lat, lon, centres, buffer_deg):
    """True if a tile centred here would share ground with a tile in `centres`.

    Every tile is a box reaching buffer_deg from its centre in each direction, so
    two boxes overlap when their centres are less than 2 * buffer_deg apart in
    BOTH latitude and longitude. Assumes the existing tiles were built with the
    same buffer_deg (0.09 for every dataset so far)."""
    sep = 2 * buffer_deg
    return any(abs(lat - a) < sep and abs(lon - o) < sep for a, o in centres)


def add_target_to_manifest(output_dir, target, last_meta, extra=None):
    """write_manifest() (from build_gee.py) hardcodes the GEE AGB collection; append the
    real target provenance to the record it just wrote. `extra` is merged into the
    same record (run_build uses it for the overlap settings)."""
    path = os.path.join(output_dir, "manifest.json")
    with open(path) as f:
        doc = json.load(f)
    rec = doc["runs"][-1]
    rec["target"] = target.describe()
    rec["agb_collection"] = None    # not from GEE
    if last_meta:
        rec["target"]["regrid"] = {k: last_meta[k] for k in ("native_res_deg", "native_shape", "ratio_tile_to_native") if k in last_meta}
    rec.update(extra or {})
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def run_build(config, target):
    """The sampling / download loop, shared by build_ect.py and build_jasmin.py.

    `target` is any object with
        fetch(bbox, year, dim) -> (y (dim,dim,1), sd (dim,dim,1) or None, meta) or None
        describe()             -> dict for the manifest
    Earth Engine must already be initialised (point sampling + AlphaEarth).

    No overlap: a random location whose tile would share ground with a known tile
    is skipped before anything is downloaded, and does not count as a try. Known
    tiles = accepted earlier in this run + already in output_dir (--append) + in
    --avoid_dirs. Before 2026-11-13 nothing was checked, so the older folders do
    contain overlapping tiles (e.g. 26 of the 45 France j300 test tiles are
    touched by a j700 train/val tile). --allow_overlap true switches the check off."""
    country_boundaries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country = country_boundaries.filter(ee.Filter.eq("country_na", config.country)).geometry()

    avoid = [d.strip() for d in (config.avoid_dirs or "").split(",") if d.strip()]
    if config.allow_overlap and avoid:
        raise SystemExit("--avoid_dirs and --allow_overlap true contradict each other: give only one")
    for d in avoid:
        # Fail loudly: a mistyped path would otherwise avoid nothing, silently.
        if not existing_tile_centres([d]):
            raise SystemExit(f"--avoid_dirs: no tiles found in {d}/ae_embeddings — check the path")
    centres = [] if config.allow_overlap else existing_tile_centres([config.output_dir] + avoid)
    if centres:
        print(f"Keeping clear of {len(centres)} existing tiles")

    accepted = tries = skipped = skipped_in_a_row = 0
    coords, last_meta = [], None
    pbar = tqdm(total=config.total_samples, desc="Accepted samples", unit="sample")
    while accepted < config.total_samples and tries < config.max_tries:
        lat, lon = sample_point_in_zone(country)
        if not config.allow_overlap and overlaps_existing(lat, lon, centres, config.buffer_deg):
            skipped += 1; skipped_in_a_row += 1
            print(f"⏭️  ({lat:.4f}, {lon:.4f}) overlaps an existing tile, skipped ({skipped} so far)")
            if skipped_in_a_row >= MAX_SKIPS_IN_A_ROW:
                print(f"\n⚠️ {MAX_SKIPS_IN_A_ROW} random locations in a row all overlapped an existing tile: "
                      f"{config.country} has no room left for more tiles. Stopping at {accepted}.")
                break
            continue
        skipped_in_a_row = 0
        tries += 1
        geom, bbox = build_geom(lat, lon, config.buffer_deg)
        print(f"\n🌍 Try {tries} | Sampling ({lat:.4f}, {lon:.4f})")
        try:
            ae = fetch_alphaearth(geom, bbox, config.master_dim, config.year)
            if ae is None:
                print("❌ AlphaEarth invalid"); continue
            res = target.fetch(bbox, config.year, config.master_dim)
            if res is None:
                print("❌ target empty"); continue
            y, sd, last_meta = res

            name = f"lat{lat:+08.4f}_lon{lon:+09.4f}_{accepted:05d}"
            np.save(os.path.join(config.ae_dir, f"{name}_ae.npy"), ae)
            np.save(os.path.join(config.target_dir, f"{name}_y.npy"), y)
            if sd is not None:
                np.save(os.path.join(config.sd_dir, f"{name}_sd.npy"), sd)
            coords.append((lat, lon))
            centres.append((lat, lon))      # later tiles in this run must stay clear of it too
            accepted += 1
            pbar.update(1); pbar.set_postfix({"tries": tries, "acc_rate": f"{accepted / tries:.2f}", "skipped": skipped})
            print(f"✅ Accepted sample {accepted}")
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    pbar.close()
    write_manifest(config, accepted, tries, coords)
    add_target_to_manifest(config.output_dir, target, last_meta, extra={
        "no_overlap": not config.allow_overlap,
        "avoid_dirs": avoid,
        "skipped_overlapping": skipped,
    })
    print(f"\nDone: {accepted} samples collected in {tries} tries ({skipped} locations skipped for overlap)")
    return accepted, tries


if __name__ == "__main__":
    ee.Initialize(project=os.getenv("EE_PROJECT", "esa-cci"))   # still needed: sampling + AlphaEarth

    config = Config(**args_extract(argparse.ArgumentParser()))
    target = EctTarget(config.data_id, config.store_id, config.var, config.sd_var, config.method, config.fill_value)
    run_build(config, target)
