"""CCI AGB targets read straight from the CEDA archive on JASMIN.

On JASMIN the whole Biomass CCI product sits on disk, so no toolbox / ODP / GEE
request is needed for the target:

    /neodc/esacci/biomass/data/agb/maps/<version>/geotiff/<year>/
        N50E000_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv6.0.tif       value, Mg/ha
        N50E000_ESACCI-BIOMASS-L4-AGB_SD-MERGED-100m-2020-fv6.0.tif    per-pixel SD

Each file is a 10 x 10 degree tile at 100 m (1/1125 deg). Versions on the archive
(checked 2026-10-20): v2.0 v3.0 v4.0 v5.0 v5.01 v6.0 v7.0. The GEE copy the
existing datasets were built from is v6, so v6.0 is the default here.

Same interface as targets_ect.EctTarget, so build_ect.run_build() accepts it:
    ArchiveTarget().fetch(bbox, year, dim) -> (y, sd, meta) or None

Resampling: nearest neighbour at tile-pixel centres (each tile pixel takes the
value of the 100 m source pixel that contains its centre). That is what GEE did
server-side and what targets_ect.regrid_to_tile does, so tiles are comparable.
A box that straddles a 10-degree tile edge is composited from the 2-4 tiles it
touches. Only numpy + rasterio; no xarray, no torch.

    python -m src.dataset.targets_cci_archive --probe 46.5 2.3
    python -m src.dataset.targets_cci_archive --backfill --data_dir <dir with targets/ + manifest.json> --n 10
    python -m src.dataset.targets_cci_archive --compare  --data_dir <same dir>
"""
import os, glob, json, time, argparse, hashlib

import numpy as np

from src.dataset.targets_ect import tile_grid, bbox_of, NAME_RE, _tiles_of, cli_compare

ARCHIVE_ROOT = "/neodc/esacci/biomass/data/agb/maps"
DEFAULT_VERSION = "v6.0"
INDEX_CACHE = os.path.expanduser("~/.cache/cci_archive")


def archive_dir(root, version, year):
    return os.path.join(root, version, "geotiff", str(year))


class ArchiveTarget:
    """One target tile (AGB + SD) from the archive GeoTIFFs."""

    def __init__(self, root=ARCHIVE_ROOT, version=DEFAULT_VERSION, fill_value=0.0, with_sd=True):
        import rasterio   # here so --help works without it
        self.rio = rasterio
        self.root, self.version = root, version
        self.fill_value, self.with_sd = fill_value, with_sd
        self._index = {}    # year -> [(agb_path, sd_path or None, (left, bottom, right, top)), ...]
        self._open = {}     # path -> open rasterio dataset (kept open: ~300 files, cheap)

    # ── index ────────────────────────────────────────────────────────────────
    def _index_year(self, year):
        """Bounds of every tile for a year. Read from the GeoTIFF headers once and
        cached as json, so the name convention (N50E000 = north-west corner) is never
        relied on."""
        if year in self._index:
            return self._index[year]
        d = archive_dir(self.root, self.version, year)
        root_tag = "" if self.root == ARCHIVE_ROOT else "_" + hashlib.md5(self.root.encode()).hexdigest()[:8]
        cache = os.path.join(INDEX_CACHE, f"{self.version}_{year}{root_tag}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                self._index[year] = [tuple(e) for e in json.load(f)]
            return self._index[year]

        files = sorted(glob.glob(os.path.join(d, "*AGB-MERGED-*.tif")))
        if not files:
            raise FileNotFoundError(f"no AGB tiles in {d} (is /neodc mounted here, and does version {self.version} have {year}?)")
        entries = []
        t0 = time.time()
        for p in files:
            with self.rio.open(p) as src:
                b = src.bounds
            sd = p.replace("AGB-MERGED", "AGB_SD-MERGED")
            entries.append((p, sd if os.path.exists(sd) else None, (b.left, b.bottom, b.right, b.top)))
        print(f"indexed {len(entries)} archive tiles for {year} in {time.time() - t0:.0f}s → {cache}")
        os.makedirs(INDEX_CACHE, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(entries, f)
        self._index[year] = entries
        return entries

    def _covering(self, year, bbox):
        min_lon, min_lat, max_lon, max_lat = bbox
        return [e for e in self._index_year(year)
                if e[2][0] < max_lon and e[2][2] > min_lon and e[2][1] < max_lat and e[2][3] > min_lat]

    def _ds(self, path):
        if path not in self._open:
            self._open[path] = self.rio.open(path)
        return self._open[path]

    # ── sampling ─────────────────────────────────────────────────────────────
    def _sample(self, path, lats, lons):
        """Nearest-neighbour read of one source tile at the given pixel centres.
        Returns (dim, dim) float32 with NaN where outside the file / nodata."""
        src = self._ds(path)
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0:
            raise ValueError(f"{path}: expected a north-up, unrotated grid, got {t}")
        cols = np.floor((lons - t.c) / t.a).astype(int)
        rows = np.floor((lats - t.f) / t.e).astype(int)       # t.e < 0: rows grow southwards
        in_c = (cols >= 0) & (cols < src.width)
        in_r = (rows >= 0) & (rows < src.height)
        out = np.full((len(lats), len(lons)), np.nan, np.float32)
        if not in_c.any() or not in_r.any():
            return out, None
        r0, r1 = rows[in_r].min(), rows[in_r].max()
        c0, c1 = cols[in_c].min(), cols[in_c].max()
        block = src.read(1, window=self.rio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(np.float32)
        if src.nodata is not None:
            block[block == src.nodata] = np.nan
        sel = np.ix_(in_r, in_c)
        rr, cc = np.meshgrid(rows[in_r] - r0, cols[in_c] - c0, indexing="ij")
        out[sel] = block[rr, cc]
        return out, {"native_res_deg": [abs(t.e), t.a], "native_shape": list(block.shape)}

    def _composite(self, paths, lats, lons):
        out, meta = None, None
        for p in paths:
            arr, m = self._sample(p, lats, lons)
            if out is None:
                out, meta = arr, m
            else:
                fill = np.isnan(out)
                out[fill] = arr[fill]
                meta = meta or m
        return out, meta

    def fetch(self, bbox, year, dim):
        """(y (dim,dim,1), sd (dim,dim,1) or None, meta) or None if the box has no
        data (same acceptance rule as the GEE and toolbox paths: mean < 1e-3)."""
        tiles = self._covering(year, bbox)
        if not tiles:
            return None
        lats, lons = tile_grid(bbox, dim)
        y, meta = self._composite([t[0] for t in tiles], lats, lons)
        if meta is None or np.all(np.isnan(y)):
            return None
        nodata = np.isnan(y)
        y = np.where(nodata, self.fill_value, y).astype(np.float32)
        if float(y.mean()) < 1e-3:
            return None
        sd = None
        sd_paths = [t[1] for t in tiles if t[1]]
        if self.with_sd and sd_paths:
            s, _ = self._composite(sd_paths, lats, lons)
            sd = np.where(np.isnan(s), self.fill_value, s).astype(np.float32)[..., None]
        meta.update(
            ratio_tile_to_native=float(abs(lats[1] - lats[0]) / meta["native_res_deg"][0]),
            nodata_frac=float(nodata.mean()), fill_value=self.fill_value, method="nearest",
            units="Mg/ha", data_id=f"{self.version}/{year}", var="AGB", sd_var="AGB_SD" if sd is not None else None,
            source_files=[os.path.basename(t[0]) for t in tiles],
        )
        return y[..., None], sd, meta

    def describe(self):
        return {"source": "ceda-archive", "root": self.root, "version": self.version,
                "var": "AGB", "sd_var": "AGB_SD" if self.with_sd else None,
                "method": "nearest", "fill_value": self.fill_value}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def cli_probe(target, lat, lon, year, buffer_deg, dim):
    bbox = bbox_of(lat, lon, buffer_deg)
    t0 = time.time()
    res = target.fetch(bbox, year, dim)
    if res is None:
        print("empty tile (no archive coverage or all zero)"); return
    y, sd, meta = res
    print(f"fetched in {time.time() - t0:.2f}s")
    print(f"AGB  mean {y.mean():.1f}  max {y.max():.0f}  zero-frac {(y == 0).mean():.3f}")
    if sd is not None:
        print(f"SD   mean {sd.mean():.1f}  max {sd.max():.0f}")
    print(json.dumps(meta, indent=2))


def cli_backfill(target, data_dir, out_subdir, n):
    """Archive targets for tiles that already exist (built from GEE), to compare."""
    run, names = _tiles_of(data_dir)
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    todo = [x for x in names if not os.path.exists(os.path.join(out_dir, f"{x}_y.npy"))][:n]
    print(f"{len(names)} tiles, fetching {len(todo)} into {out_dir}/")
    times, failed = [], []
    for name in todo:
        m = NAME_RE.match(name)
        bbox = bbox_of(float(m.group(1)), float(m.group(2)), run["buffer_deg"])
        t0 = time.time()
        res = target.fetch(bbox, run["year"], run["master_dim"])
        times.append(time.time() - t0)
        if res is None:
            print(f"  {name}: empty"); failed.append(name); continue
        y, sd, _ = res
        np.save(os.path.join(out_dir, f"{name}_y.npy"), y)
        if sd is not None:
            np.save(os.path.join(out_dir, f"{name}_sd.npy"), sd)
    if times:
        print(f"done: {len(times) - len(failed)} written, {len(failed)} failed, {np.mean(times):.2f}s/tile")
    with open(os.path.join(out_dir, "source.json"), "w") as f:
        json.dump(target.describe(), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ARCHIVE_ROOT)
    p.add_argument("--version", default=DEFAULT_VERSION, help="archive version dir, e.g. v6.0 or v7.0")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--probe", nargs=2, type=float, metavar=("LAT", "LON"), help="fetch one tile and print stats")
    p.add_argument("--buffer_deg", type=float, default=0.09)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--backfill", action="store_true", help="archive targets for the tiles in --data_dir")
    p.add_argument("--compare", action="store_true", help="compare targets/ with --subdir in --data_dir")
    p.add_argument("--data_dir"); p.add_argument("--subdir", default="targets_archive")
    p.add_argument("--n", type=int, default=10**9)
    a = p.parse_args()

    if a.probe:
        cli_probe(ArchiveTarget(a.root, a.version), a.probe[0], a.probe[1], a.year, a.buffer_deg, a.dim)
    elif a.backfill:
        cli_backfill(ArchiveTarget(a.root, a.version), a.data_dir, a.subdir, a.n)
    elif a.compare:
        cli_compare(a.data_dir, a.subdir)
    else:
        p.print_help()
