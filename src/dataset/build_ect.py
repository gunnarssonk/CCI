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
"""
import os, json, time, argparse
from dataclasses import dataclass, fields

import ee
import numpy as np
from tqdm import tqdm

from src.dataset.build_gee import build_geom, fetch_alphaearth, sample_point_in_zone, write_manifest, str2bool
from src.dataset.targets_ect import EctTarget, DEFAULT_DATA_ID, DEFAULT_STORE


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

    def __post_init__(self):
        if not self.output_dir:
            raise SystemExit("--output_dir is required, e.g. --output_dir data_gee/data_france_2020_ect")
        self.ae_dir = os.path.join(self.output_dir, "ae_embeddings")
        self.target_dir = os.path.join(self.output_dir, "targets")
        self.sd_dir = os.path.join(self.output_dir, "targets_sd")
        for d in (self.output_dir, self.ae_dir, self.target_dir, self.sd_dir):
            os.makedirs(d, exist_ok=True)


def args_extract(parser):
    for f in fields(Config):
        t = str2bool if f.type is bool else f.type
        parser.add_argument(f"--{f.name}", type=t, default=f.default)
    return vars(parser.parse_args())


def add_target_to_manifest(output_dir, target, last_meta):
    """write_manifest() (from build_gee.py) hardcodes the GEE AGB collection; append the
    real target provenance to the record it just wrote."""
    path = os.path.join(output_dir, "manifest.json")
    with open(path) as f:
        doc = json.load(f)
    rec = doc["runs"][-1]
    rec["target"] = target.describe()
    rec["agb_collection"] = None    # not from GEE
    if last_meta:
        rec["target"]["regrid"] = {k: last_meta[k] for k in ("native_res_deg", "native_shape", "ratio_tile_to_native") if k in last_meta}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


if __name__ == "__main__":
    ee.Initialize(project=os.getenv("EE_PROJECT", "esa-cci"))   # still needed: sampling + AlphaEarth

    config = Config(**args_extract(argparse.ArgumentParser()))
    target = EctTarget(config.data_id, config.store_id, config.var, config.sd_var, config.method, config.fill_value)

    country_boundaries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country = country_boundaries.filter(ee.Filter.eq("country_na", config.country)).geometry()

    accepted = tries = 0
    coords, last_meta = [], None
    pbar = tqdm(total=config.total_samples, desc="Accepted samples", unit="sample")
    while accepted < config.total_samples and tries < config.max_tries:
        tries += 1
        lat, lon = sample_point_in_zone(country)
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
            accepted += 1
            pbar.update(1); pbar.set_postfix({"tries": tries, "acc_rate": f"{accepted / tries:.2f}"})
            print(f"✅ Accepted sample {accepted}")
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Error: {e}")

    pbar.close()
    write_manifest(config, accepted, tries, coords)
    add_target_to_manifest(config.output_dir, target, last_meta)
    print(f"\nDone: {accepted} samples collected in {tries} tries")
