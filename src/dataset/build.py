#!/Users/klara.gunnarsson/miniforge3/envs/esa_env/bin/python
import argparse
from dataclasses import dataclass, fields
import json
import os
import subprocess
import io
import sys
import time
import random
import requests
import numpy as np
import tifffile
import ee

from urllib.error import HTTPError
from rasterio.transform import from_bounds
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

@dataclass
class Config:
    """Configuration for the dataset to build."""
    # No default on purpose. A default output_dir once caused two Sweden runs to be
    # appended into "data_uniform" because --output_dir was forgotten, leaving a
    # folder that was 2/3 Swedish while still being treated as French data.
    output_dir: str = None
    master_dim: int = 256
    buffer_deg: float = 0.09  # ~10km
    total_samples: int = 100
    max_tries: int = 100000  # safety caps
    year: int = 2020
    country: str = "France"

    def __post_init__(self):
        if not self.output_dir:
            raise SystemExit(
                "--output_dir is required. Name it for what it will contain, "
                "e.g. --output_dir data_gee/data_france_2020"
            )
        self.ae_dir: str = os.path.join(self.output_dir, "ae_embeddings")
        self.target_dir: str = os.path.join(self.output_dir, "targets")
        self.sd_dir: str = os.path.join(self.output_dir, "targets_sd")   # CCI per-pixel uncertainty
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ae_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.sd_dir, exist_ok=True)


def str2bool(value):
    """argparse would read the string "False" as True, so parse booleans by hand."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def args_extract(parser: argparse.ArgumentParser):
    for field in fields(Config):
        # Determine the type (handling types like 'type' carefully)
        field_type = field.type if field.type != type else None
        if field_type is bool:
            field_type = str2bool
        parser.add_argument(
            f"--{field.name}",
            type=field_type,
            default=field.default
        )

    args = parser.parse_args()
    config_keys = {f.name for f in fields(Config)}
    extra_args = set(vars(args).keys()) - config_keys
    if extra_args:
        print(f"Arguments ignored (not in Config): {', '.join(extra_args)}")

    filtered_args = {k: v for k, v in vars(args).items() if k in config_keys}
    return filtered_args


# ==============================
# UTIL FUNCTIONS
# ==============================

def sample_uniform_lat_lon():
    """Uniform sampling on sphere"""
    u = random.random()
    v = random.random()

    lat = np.degrees(np.arcsin(2 * u - 1))
    lon = 360 * v - 180
    return lat, lon

def is_on_land(lat, lon, land_geom):
    pt = ee.Geometry.Point([lon, lat])
    try:
        return land_geom.contains(pt).getInfo()
    except Exception:
        return False
    

def sample_point_in_zone(france_geom):
    pt = ee.FeatureCollection.randomPoints(
        region=france_geom,
        points=1,
        seed=random.randint(0, 1_000_000)
    ).first()

    coords = pt.geometry().coordinates().getInfo()
    lon, lat = coords
    return lat, lon







def build_geom(lat, lon, buffer_deg):
    min_lon = lon - buffer_deg
    max_lon = lon + buffer_deg
    min_lat = lat - buffer_deg
    max_lat = lat + buffer_deg

    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat]), \
           (min_lon, min_lat, max_lon, max_lat) # the plain tuple which is the format the CCI toolbox want for region= / this line is an integration point later

# download ae input data from google earth engine and return as numpy array
def fetch_alphaearth(geom, bounds, master_dim, year):
    ae_coll = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate(f'{year}-01-01', f'{year+1}-01-01')
               .filterBounds(geom)
               .sort('system:time_start', False))

    # The annual collection is a patchwork of many images, each with its own footprint.
    # .first() used to pick a single one, so any box straddling a footprint edge came
    # back partly as no-data fill (16 of 100 France-2020 tiles, ~7% of all pixels, one
    # tile fully empty). mosaic() stitches every image covering the box instead.
    if ae_coll.size().getInfo() == 0:
        return None

    ae_img = ae_coll.mosaic().unmask(0).toFloat()

    bands = [f"A{i:02d}" for i in range(64)]
    chunk_size = 64
    chunks = []

    for i in range(0, 64, chunk_size):
        sub = bands[i:i+chunk_size]
        chunk_img = ae_img.select(sub)

        url = chunk_img.getDownloadURL({
            'region': geom,
            'format': 'GEO_TIFF',
            'dimensions': f'{master_dim}x{master_dim}',
            'crs': 'EPSG:4326'
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as f:
            arr = tifffile.imread(f)
            arr = np.nan_to_num(arr, nan=0.0)

            if arr.ndim == 2:
                arr = arr[..., None]

            chunks.append(arr)

    ae = np.concatenate(chunks, axis=-1)

    # quality check
    if np.mean(np.abs(ae)) < 1e-6:
        return None

    return ae

# give the function a year, download the target data from google earth engine and return as numpy array
def fetch_agb(geom, year, master_dim, with_sd=False):
    """Download the CCI biomass tile. Returns (H, W, 1) AGB in Mg/ha, or with
    with_sd=True a tuple (agb, sd) where sd is CCI's own per-pixel standard
    deviation. Both bands come from ONE request so they are pixel-aligned: two
    separate downloads of the same box can differ on ~3% of pixels, because the
    100 m source is resampled to the tile grid with nearest neighbour and edge
    pixels flip between neighbours."""
    img = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filterBounds(geom)
           .first())

    if img is None:
        return None

    img = img.select(['AGB', 'SD'] if with_sd else ['AGB'])

    url = img.getDownloadURL({
        'region': geom,
        'format': 'GEO_TIFF',
        'dimensions': f'{master_dim}x{master_dim}',
        'crs': 'EPSG:4326'
    })

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with io.BytesIO(resp.content) as f:
        arr = tifffile.imread(f)
        arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)

        if arr.ndim == 2:
            arr = arr[..., None]

    agb = arr[..., :1]

    # quality check
    if np.mean(agb) < 1e-3:
        return None

    if with_sd:
        return agb, arr[..., 1:2]
    return agb



def write_manifest(config, accepted, tries, coords):
    """Append a record of this run to <output_dir>/manifest.json.

    build.py appends to whatever directory it is given, so one folder can hold
    tiles from several runs -- possibly with different countries, years or code
    versions. Nothing in the .npy files themselves records any of that, so a
    mixed folder is invisible until someone parses coordinates out of filenames.
    This manifest makes each run's provenance explicit.
    """
    path = os.path.join(config.output_dir, "manifest.json")

    runs = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                runs = json.load(f).get("runs", [])
        except (json.JSONDecodeError, OSError):
            print("   ⚠️ existing manifest unreadable, starting a new one")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    record = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "country": config.country,
        "year": config.year,
        "requested": config.total_samples,
        "accepted": accepted,
        "tries": tries,
        "buffer_deg": config.buffer_deg,
        "master_dim": config.master_dim,
        "git_commit": commit,
        "ae_collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "agb_collection": "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB",
    }
    if coords:
        lats = [c[0] for c in coords]; lons = [c[1] for c in coords]
        record["lat_range"] = [round(min(lats), 4), round(max(lats), 4)]
        record["lon_range"] = [round(min(lons), 4), round(max(lons), 4)]

    runs.append(record)
    with open(path, "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"📝 manifest updated → {path} ({len(runs)} run(s) recorded)")


# ==============================
# MAIN LOOP
# a lot of things are hardcoded - when running enter on command line country and samples - make it wasier to change sample and location 
# evaluate model on a complete different area and time period 
# ==============================
if __name__ == "__main__":
    project = os.getenv("EE_PROJECT")
    if project:
        ee.Initialize(project=project)
    else:
        print("EE_PROJECT env var not set, using default project")
        ee.Initialize(project="esa-cci")  # fallback "alexcloud-489214"

    parser = argparse.ArgumentParser()
    filtered_args = args_extract(parser)
    config = Config(**filtered_args)

    country_boundaries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") 
    land_geom = country_boundaries.geometry()
    country = country_boundaries.filter(ee.Filter.eq('country_na', config.country)).geometry()
    accepted = 0 
    tries = 0
    coords = []   # lat/lon of accepted tiles, recorded in the manifest

    pbar = tqdm(total=config.total_samples, desc="Accepted samples", unit="sample")
    while accepted < config.total_samples and tries < config.max_tries:
        tries += 1

        lat, lon = sample_point_in_zone(country)
        geom, bounds = build_geom(lat, lon, config.buffer_deg)

        print(f"\n🌍 Try {tries} | Sampling ({lat:.4f}, {lon:.4f})")

        try:
            ae = fetch_alphaearth(geom, bounds, config.master_dim, config.year)
            if ae is None:
                print("❌ AlphaEarth invalid")
                continue

            fetched = fetch_agb(geom, config.year, config.master_dim, with_sd=True)
            if fetched is None:
                print("❌ AGB invalid")
                continue
            agb, sd = fetched

            # SAVE
            #name = f"sample_{accepted:05d}"
            #name = f"lat{lat:.4f}_lon{lon:.4f}"
            name = f"lat{lat:+08.4f}_lon{lon:+09.4f}_{accepted:05d}"

            np.save(os.path.join(config.ae_dir, f"{name}_ae.npy"), ae)
            np.save(os.path.join(config.target_dir, f"{name}_y.npy"), agb)
            np.save(os.path.join(config.sd_dir, f"{name}_sd.npy"), sd)
            coords.append((lat, lon))

            #print(f"✅ Accepted sample {accepted}")
            accepted += 1
            pbar.update(1)
            accept_rate = accepted / tries
            pbar.set_postfix({
                "tries": tries,
                "acc_rate": f"{accept_rate:.2f}"
            })
            print(f"✅ Accepted sample {accepted}")

            # small delay to avoid throttling
            time.sleep(1)

        except HTTPError:
            print("   ⚠️ HTTP error")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    pbar.close()
    write_manifest(config, accepted, tries, coords)
    print("\n====================")
    print(f"Done: {accepted} samples collected in {tries} tries")