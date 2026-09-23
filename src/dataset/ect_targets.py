"""Target tiles from the ESA Climate Toolbox (CCI Open Data Portal) instead of
Google Earth Engine.

Standalone: nothing else in the project imports this module, and it never imports
torch. It needs the `ect` environment (setup/default/ect_env.sh), because the
toolbox depends on GDAL which pip cannot install into .venv.

    mamba activate ect
    python -m src.dataset.ect_targets --list-ecvs
    python -m src.dataset.ect_targets --ecv BIOMASS
    python -m src.dataset.ect_targets --data_id esacci.BIOMASS.yr.L4.AGB.multi-sensor.multi-platform.MERGED.6-0.100m --store esa-cci
    python -m src.dataset.ect_targets --backfill --data_dir data_gee/data_france_2020_v2 --n 10
    python -m src.dataset.ect_targets --compare  --data_dir data_gee/data_france_2020_v2

How it differs from the GEE path: Earth Engine resampled every layer to the tile
grid (dim x dim over the bbox) server-side. The toolbox returns the dataset on its
native grid (100 m for AGB), so `regrid_to_tile` resamples onto the same tile grid
here, nearest-neighbour at pixel centres. Expect ~97% pixel-exact agreement with
the GEE tiles; the rest is resampling jitter at sharp edges.

Limit: the tile approach needs the ECV's native resolution to be well below the
tile size. A 0.18 deg tile of a 0.25 deg product is less than one native pixel;
`regrid_to_tile` refuses that rather than silently producing a constant tile.
"""
import os, re, json, time, argparse, warnings

import numpy as np

# Toolbox / xcube imports are inside the functions that need them, so that
# --help and the pure-numpy helpers work in any environment.

# Default: the Zarr store. It is fast (~0.6 s per tile) and works. The ODP store
# ("esa-cci", which has AGB v5/v6/v7) fails for the AGB datasets with toolbox
# 1.7.2 / xcube-cci 0.13: CciChunkStore._adjust_coord_data crashes on the
# `lat_bnds` coordinate (list instead of array, then a chunk-size mismatch).
# Tested 2026-09-23. So the toolbox path currently gives AGB v4 (2010-2020),
# while the GEE tiles are v6 — versions differ, expect real differences.
DEFAULT_DATA_ID = "ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-2020-fv4.0.zarr"
DEFAULT_STORE = "esa-cci-zarr"
ODP_AGB_V6 = "esacci.BIOMASS.yr.L4.AGB.multi-sensor.multi-platform.MERGED.6-0.100m"   # broken on ODP, see above
NAME_RE = re.compile(r"lat([+-]\d+\.\d+)_lon([+-]\d+\.\d+)_(\d+)")


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers (numpy + xarray only)
# ─────────────────────────────────────────────────────────────────────────────
def bbox_of(lat, lon, buffer_deg):
    """Same box as build.build_geom() returns as its second element, without
    needing an initialised Earth Engine client."""
    return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)


def tile_grid(bbox, dim):
    """Pixel centres of the dim x dim tile GEE produced for `dimensions` over bbox.
    lons run west->east, lats north->south (row 0 = north edge, as in the .npy tiles)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon = (max_lon - min_lon) / dim
    dlat = (max_lat - min_lat) / dim
    lons = min_lon + (np.arange(dim) + 0.5) * dlon
    lats = max_lat - (np.arange(dim) + 0.5) * dlat
    return lats, lons


def _coord_names(da):
    lat = next((c for c in da.coords if c.lower() in ("lat", "latitude", "y")), None)
    lon = next((c for c in da.coords if c.lower() in ("lon", "longitude", "x")), None)
    if lat is None or lon is None:
        raise ValueError(f"cannot find lat/lon coords in {list(da.coords)}")
    return lat, lon


def regrid_to_tile(da, lats, lons, method="nearest", fill_value=0.0):
    """Resample an xarray DataArray (native grid, bbox subset) onto the tile grid.
    Returns (array (dim, dim) float32, meta dict)."""
    da = da.squeeze(drop=True)
    if da.ndim != 2:
        raise ValueError(f"expected a single 2-D field after squeezing, got dims {da.dims}")
    lat_n, lon_n = _coord_names(da)
    da = da.sortby(lat_n).sortby(lon_n)          # interp needs ascending coords

    native_lat = float(np.median(np.abs(np.diff(da[lat_n].values))))
    native_lon = float(np.median(np.abs(np.diff(da[lon_n].values))))
    n_native = (int(da.sizes[lat_n]), int(da.sizes[lon_n]))
    if min(n_native) < 2:
        raise ValueError(
            f"bbox covers only {n_native} native pixels (native res ~{native_lat:.4f} deg): "
            f"this ECV is too coarse for the tile approach")
    if min(n_native) < len(lats) / 8:
        warnings.warn(f"only {n_native} native pixels for a {len(lats)}x{len(lons)} tile; heavy upsampling")

    out = da.interp({lat_n: lats, lon_n: lons}, method=method).values.astype(np.float32)
    nodata = ~np.isfinite(out)
    out = np.where(nodata, fill_value, out)
    meta = {
        "native_res_deg": [native_lat, native_lon],
        "native_shape": list(n_native),
        "ratio_tile_to_native": float(abs(lats[1] - lats[0]) / native_lat),
        "nodata_frac": float(nodata.mean()),
        "fill_value": fill_value,
        "method": method,
    }
    return out, meta


# ─────────────────────────────────────────────────────────────────────────────
# Toolbox access
# ─────────────────────────────────────────────────────────────────────────────
class EctTarget:
    """Fetches one target tile (value + uncertainty) from a CCI dataset."""

    def __init__(self, data_id=DEFAULT_DATA_ID, store_id=DEFAULT_STORE, var=None, sd_var=None,
                 method="nearest", fill_value=0.0, pad_deg=0.01):
        from xcube.core.store import new_data_store
        self.store_id, self.data_id = store_id, data_id
        self.store = new_data_store(store_id)
        self.var, self.sd_var = var, sd_var
        self.method, self.fill_value, self.pad = method, fill_value, pad_deg
        self._resolved = var is not None

    def _resolve_vars(self, ds):
        """Pick the value / uncertainty variables if not given explicitly."""
        names = [v for v in ds.data_vars if ds[v].ndim >= 2]
        def is_unc(v):   # uncertainty-like names: agb_sd, agb_se, *_std, *_err, *_uncertainty
            s = v.lower()
            return any(s.endswith(x) for x in ("_sd", "_se", "_std", "_err", "_error", "_uncertainty", "_unc"))
        if self.var is None:
            cands = [v for v in names if not is_unc(v)]
            if len(cands) != 1:
                raise ValueError(f"cannot pick the value variable from {names}; pass --var")
            self.var = cands[0]
        if self.sd_var is None:
            cands = [v for v in names if v != self.var and is_unc(v)]
            self.sd_var = cands[0] if len(cands) == 1 else None
        self._resolved = True
        print(f"ECT variables: value={self.var!r}, sd={self.sd_var!r}  (from {names})")

    def open(self, bbox, year):
        """Lazy xarray Dataset covering bbox (plus padding) for one year.

        Two kinds of store: the ODP store ('esa-cci') subsets server-side via
        bbox/time_range. The 'esa-cci-zarr' and 'esa-cci-kc' stores take no bbox;
        the whole (lazy) dataset is opened once and sliced per tile, which is
        fast (~0.6 s per tile for AGB)."""
        p = self.pad
        padded = (bbox[0] - p, bbox[1] - p, bbox[2] + p, bbox[3] + p)
        if self.store_id == "esa-cci":
            return self.store.open_data(self.data_id, bbox=padded,
                                        time_range=(f"{year}-01-01", f"{year}-12-31"))
        if not hasattr(self, "_full"):
            self._full = self.store.open_data(self.data_id)
        ds = self._full
        lat_desc = float(ds.lat[0]) > float(ds.lat[-1])
        lat_slice = slice(padded[3], padded[1]) if lat_desc else slice(padded[1], padded[3])
        sub = ds.sel(lat=lat_slice, lon=slice(padded[0], padded[2]))
        if "time" in sub.dims:
            sub = sub.sel(time=str(year))
        return sub

    def fetch(self, bbox, year, dim):
        """Returns (y (dim,dim,1), sd (dim,dim,1) or None, meta) or None if the tile
        is empty (same acceptance rule as the GEE path)."""
        ds = self.open(bbox, year)
        if not self._resolved:
            self._resolve_vars(ds)
        if "time" in ds.dims and ds.sizes["time"] != 1:
            raise ValueError(f"{ds.sizes['time']} time steps in {year} for {self.data_id}; expected 1")

        lats, lons = tile_grid(bbox, dim)
        y, meta = regrid_to_tile(ds[self.var], lats, lons, self.method, self.fill_value)
        if float(y.mean()) < 1e-3:
            return None
        sd = None
        if self.sd_var:
            sd = regrid_to_tile(ds[self.sd_var], lats, lons, self.method, self.fill_value)[0][..., None]
        meta.update(units=ds[self.var].attrs.get("units"), data_id=self.data_id, var=self.var, sd_var=self.sd_var)
        return y[..., None], sd, meta

    def describe(self):
        try:
            import esa_climate_toolbox
            ver = getattr(esa_climate_toolbox, "__version__", "?")
        except ImportError:
            ver = "?"
        return {"source": "esa-climate-toolbox", "toolbox_version": ver, "store": self.store_id,
                "data_id": self.data_id, "var": self.var, "sd_var": self.sd_var,
                "method": self.method, "fill_value": self.fill_value}


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────
def cli_list_ecvs():
    from esa_climate_toolbox.core import list_ecvs
    for e in list_ecvs():
        print(e)


def cli_list_datasets(ecv):
    from esa_climate_toolbox.core import list_ecv_datasets
    rows = list_ecv_datasets(ecv)
    for data_id, store_id in rows:
        print(f"{store_id:<14} {data_id}")
    print(f"\n{len(rows)} datasets for {ecv}")


def cli_describe(data_id, store_id):
    from xcube.core.store import new_data_store
    store = new_data_store(store_id)
    d = store.describe_data(data_id)
    info = d.to_dict()
    print(json.dumps({k: info[k] for k in ("data_id", "bbox", "spatial_res", "time_range", "time_period", "dims") if k in info}, indent=2, default=str))
    print("variables:")
    for name, v in (info.get("data_vars") or {}).items():
        print(f"  {name:<12} dims={v.get('dims')}  dtype={v.get('dtype')}  {v.get('attrs', {}).get('long_name', '')}")


def _tiles_of(data_dir):
    with open(os.path.join(data_dir, "manifest.json")) as f:
        run = json.load(f)["runs"][-1]
    names = sorted(f[:-len("_y.npy")] for f in os.listdir(os.path.join(data_dir, "targets")) if f.endswith("_y.npy"))
    return run, names


def cli_backfill(data_dir, out_subdir, n, target):
    """Fetch ECT targets for tiles that already exist (from GEE), for comparison."""
    run, names = _tiles_of(data_dir)
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    todo = [x for x in names if not os.path.exists(os.path.join(out_dir, f"{x}_y.npy"))][:n]
    print(f"{len(names)} tiles, fetching {len(todo)} into {out_dir}/")
    times, failed, meta = [], [], None
    for name in todo:
        m = NAME_RE.match(name)
        lat, lon = float(m.group(1)), float(m.group(2))
        bbox = bbox_of(lat, lon, run["buffer_deg"])
        t0 = time.time()
        try:
            res = target.fetch(bbox, run["year"], run["master_dim"])
        except Exception as e:
            print(f"  {name}: {e}"); failed.append(name); continue
        times.append(time.time() - t0)
        if res is None:
            print(f"  {name}: empty tile from ECT"); failed.append(name); continue
        y, sd, meta = res
        np.save(os.path.join(out_dir, f"{name}_y.npy"), y)
        if sd is not None:
            np.save(os.path.join(out_dir, f"{name}_sd.npy"), sd)
        print(f"  {name}  {times[-1]:.1f}s")
    if times:
        print(f"done: {len(times)} written, {len(failed)} failed, {np.mean(times):.1f}s/tile")
    if meta:
        print("regrid meta (last tile):", json.dumps({k: meta[k] for k in ("native_shape", "native_res_deg", "ratio_tile_to_native", "nodata_frac")}))
    with open(os.path.join(out_dir, "source.json"), "w") as f:
        json.dump(target.describe(), f, indent=2)


def cli_compare(data_dir, ect_subdir):
    """Pixel agreement between GEE targets and ECT targets for the same tiles."""
    _, names = _tiles_of(data_dir)
    ect_dir = os.path.join(data_dir, ect_subdir)
    rows = []
    for name in names:
        p_ect = os.path.join(ect_dir, f"{name}_y.npy")
        if not os.path.exists(p_ect):
            continue
        a = np.load(os.path.join(data_dir, "targets", f"{name}_y.npy")).squeeze()
        b = np.load(p_ect).squeeze()
        d = np.abs(a - b)
        row = {"tile": name, "frac_equal": float((a == b).mean()), "mean_abs": float(d.mean()),
               "p99_abs": float(np.percentile(d, 99)), "max_abs": float(d.max()),
               "gee_mean": float(a.mean()), "ect_mean": float(b.mean())}
        p_sd_gee = os.path.join(data_dir, "targets_sd", f"{name}_sd.npy")
        p_sd_ect = os.path.join(ect_dir, f"{name}_sd.npy")
        if os.path.exists(p_sd_gee) and os.path.exists(p_sd_ect):
            s1 = np.load(p_sd_gee).squeeze(); s2 = np.load(p_sd_ect).squeeze()
            row["sd_frac_equal"] = float((s1 == s2).mean())
        rows.append(row)
    if not rows:
        print("no tiles with both GEE and ECT targets"); return
    print(f"{'tile':<34}{'equal':>7}{'meanΔ':>8}{'p99Δ':>7}{'maxΔ':>7}{'gee':>7}{'ect':>7}{'sd=':>7}")
    for r in rows:
        print(f"{r['tile']:<34}{r['frac_equal']:>7.3f}{r['mean_abs']:>8.2f}{r['p99_abs']:>7.1f}{r['max_abs']:>7.0f}"
              f"{r['gee_mean']:>7.1f}{r['ect_mean']:>7.1f}{r.get('sd_frac_equal', float('nan')):>7.3f}")
    fe = np.array([r["frac_equal"] for r in rows])
    print(f"\n{len(rows)} tiles: pixel-exact agreement mean {fe.mean():.3f} (min {fe.min():.3f}), "
          f"mean |diff| {np.mean([r['mean_abs'] for r in rows]):.2f} Mg/ha")
    if fe.mean() < 0.9:
        print("LOW agreement: suspect a grid offset in tile_grid (half pixel or row order).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-ecvs", action="store_true")
    p.add_argument("--ecv", help="list datasets for this ECV, e.g. BIOMASS")
    p.add_argument("--data_id", default=None, help="describe this dataset (with no other action) / use it for backfill")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--var", default=None); p.add_argument("--sd_var", default=None)
    p.add_argument("--method", default="nearest", choices=["nearest", "linear"])
    p.add_argument("--backfill", action="store_true", help="fetch ECT targets for the tiles in --data_dir")
    p.add_argument("--compare", action="store_true", help="compare targets/ with --ect_subdir in --data_dir")
    p.add_argument("--data_dir"); p.add_argument("--ect_subdir", default="targets_ect")
    p.add_argument("--n", type=int, default=10**9, help="max tiles to backfill")
    a = p.parse_args()

    if a.list_ecvs:
        cli_list_ecvs()
    elif a.ecv:
        cli_list_datasets(a.ecv)
    elif a.backfill:
        t = EctTarget(a.data_id or DEFAULT_DATA_ID, a.store, a.var, a.sd_var, a.method)
        cli_backfill(a.data_dir, a.ect_subdir, a.n, t)
    elif a.compare:
        cli_compare(a.data_dir, a.ect_subdir)
    elif a.data_id:
        cli_describe(a.data_id, a.store)
    else:
        p.print_help()
