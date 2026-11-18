"""Shared pieces of the v7 setup: the six boxes, the CCI pixel grid, the blocks,
and the run record every v7 script writes next to its output.

v7 setup (decided 2026-11-18):
  target  CCI Biomass v7.0, year 2020, read from the archive on JASMIN
  input   AlphaEarth embeddings from Earth Engine, averaged onto the CCI grid
  ground  six boxes of 1 x 1 degree, two per biome
  blocks  each box is cut into 5 x 5 blocks of 225 x 225 CCI pixels

Words: a *box* is one 1-degree square; a *block* is one of its 25 pieces; a
*pixel* is one CCI pixel (1/1125 degree, about 100 m north-south).

Nothing in src/v7 imports from, or writes into, the v6 data, checkpoints or results.
"""
import os, sys, json, math, time, socket, subprocess
from fractions import Fraction

ARCHIVE_DIR = "/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/2020"
CCI_VERSION = "v7.0"
YEAR = 2020
PIX_PER_DEG = 1125                  # CCI pixel = 1/1125 degree
BOX_PIX = 1125                      # a box is 1 degree = 1125 x 1125 pixels
BLOCK_PIX = 225                     # 1125 / 5: 5 x 5 = 25 blocks per box
BLOCKS_PER_SIDE = BOX_PIX // BLOCK_PIX
DATA_ROOT = "/gws/ssde/j25a/esaclim/klara/v7"

# name: (south, north, west, east, biome), exactly as given by Klara on 2026-11-18
BOXES = {
    "central_sweden":   ("61.00", "62.00", "16.00", "17.00", "boreal"),
    "northern_sweden":  ("63.00", "64.00", "16.00", "17.00", "boreal"),
    "netherlands":      ("51.50", "52.50", "5.00", "6.00", "temperate"),
    "southwest_sweden": ("57.00", "58.00", "12.50", "13.50", "temperate"),
    "amapa_brazil":     ("0.75", "1.75", "-52.25", "-51.25", "tropical moist"),
    "east_para_brazil": ("-4.00", "-3.00", "-49.00", "-48.00", "tropical moist"),
}


def _nearest_tie_down(x):
    """Nearest whole number; an exact half goes down (towards north / west)."""
    return math.ceil(x - Fraction(1, 2))


def snap(box):
    """Where a box sits on the CCI pixel grid.

    The CCI files are 10 x 10 degree squares named after their north-west corner.
    The box's north and west edges are moved to the nearest pixel edge (an edge
    exactly in the middle of a pixel goes north / west), and the box is then
    exactly BOX_PIX x BOX_PIX whole pixels. Fractions keep this exact: with
    ordinary decimals 8437.5 could come out as 8437.4999.
    """
    south, north, west, east, biome = BOXES[box]
    south, north, west, east = (Fraction(v) for v in (south, north, west, east))
    assert north - south == 1 and east - west == 1, f"{box}: not a 1 x 1 degree box"
    lat_top = 10 * math.ceil(north / 10)          # north edge of the 10-degree CCI file
    lon_left = 10 * math.floor(west / 10)         # west edge of the 10-degree CCI file
    row0 = _nearest_tie_down((lat_top - north) * PIX_PER_DEG)
    col0 = _nearest_tie_down((west - lon_left) * PIX_PER_DEG)
    north_s = lat_top - Fraction(row0, PIX_PER_DEG)
    west_s = lon_left + Fraction(col0, PIX_PER_DEG)
    tile = f"{'N' if lat_top >= 0 else 'S'}{abs(lat_top):02d}{'E' if lon_left >= 0 else 'W'}{abs(lon_left):03d}"
    return {
        "box": box, "biome": biome, "tile": tile, "lat_top": lat_top, "lon_left": lon_left,
        "row0": row0, "col0": col0,
        "requested": {"south": float(south), "north": float(north), "west": float(west), "east": float(east)},
        "snapped": {"south": float(north_s - 1), "north": float(north_s), "west": float(west_s), "east": float(west_s + 1)},
        "shift_north_pixels": float((north_s - north) * PIX_PER_DEG),     # + = moved north
        "shift_east_pixels": float((west_s - west) * PIX_PER_DEG),        # + = moved east
    }


def cci_path(tile, kind):
    """kind: 'AGB' (biomass, Mg/ha) or 'AGB_SD' (CCI's own per-pixel uncertainty)."""
    return os.path.join(ARCHIVE_DIR, f"{tile}_ESACCI-BIOMASS-L4-{kind}-MERGED-100m-{YEAR}-f{CCI_VERSION}.tif")


def crop_path(box, kind):
    return os.path.join(DATA_ROOT, "cci", f"{box}_{kind}_{YEAR}_f{CCI_VERSION}.tif")


def blocks():
    """(name, row, col) of the 25 blocks of a box; row 0 is the northern row, col 0 the western column."""
    return [(f"r{r}_c{c}", r, c) for r in range(BLOCKS_PER_SIDE) for c in range(BLOCKS_PER_SIDE)]


def pixel_size_m(lat_deg):
    """(north-south, east-west) size of one CCI pixel on the ground, in metres, at this latitude."""
    p = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    m_per_deg_lon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)
    return m_per_deg_lat / PIX_PER_DEG, m_per_deg_lon / PIX_PER_DEG


def write_run_record(out_dir, name, settings=None, results=None):
    """Save the exact command, all settings and where/when it ran, next to the output.
    One file per run (the Slurm job number or the time is in the name), so nothing is overwritten."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        # any changed or new (not yet committed) file under src/ or setup/: the code then differs from git_commit
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "setup"], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        commit, dirty = "unknown", None
    job = os.getenv("SLURM_JOB_ID")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    module = sys.modules["__main__"].__spec__.name if getattr(sys.modules["__main__"], "__spec__", None) else sys.argv[0]
    record = {
        "command": " ".join(["python", "-m", module] + sys.argv[1:]),
        "working_dir": os.getcwd(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "slurm_job_id": job,
        "git_commit": commit,
        "code_differs_from_git_commit": dirty,
        "python": sys.version.split()[0],
        "settings": settings or {},
        "results": results or {},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{name}_{'job' + job if job else stamp}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"run record -> {path}")
    return path
