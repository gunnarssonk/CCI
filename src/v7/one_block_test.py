"""The one-block test: does Earth Engine deliver averaged embeddings on the CCI grid?

    python -m src.v7.one_block_test --box netherlands --block r2_c3

Asks Earth Engine for one block (225 x 225 CCI pixels) and checks three things:

  1. Grid.   The delivered block sits exactly on the CCI pixel grid.
  2. Length. The delivered embeddings are shorter than 1 on mixed ground
             (a single 10 m embedding has length 1; an average of different ones is shorter).
  3. Average. For a small patch (40 x 40 CCI pixels) the original 10 m embeddings are
             fetched too and averaged here, independently, with numpy. Earth Engine's
             result must match this average, and must NOT look like a single 10 m spot.

Two further numbers, for information:
  - shift test: the embeddings explain CCI biomass best when the two maps are not
    shifted against each other (a ridge regression is fitted for shifts of -2..+2 pixels);
  - averaged against single-spot input: how well each explains CCI biomass on this block.

Thresholds were fixed before the first run:
  grid     transform equal to CCI's within 1e-9 degree, 225 x 225, EPSG:4326
  length   Earth Engine length agrees with the length of my own average
           (mean difference below 0.005) and single 10 m pixels have length 0.98-1.02
  average  RMS difference to my own average below 0.005 (a typical band value is 0.125)
           and below one fifth of the RMS difference to a single spot

Outputs: numbers and the figure in results/v7/one_block_test/<box>_<block>_job<id>/,
the delivered arrays in <DATA_ROOT>/one_block_test/<same name>/. Nothing existing is overwritten.
"""
import os, json, time, argparse

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_points, transform_bounds

from src.v7 import ee_embeddings as E
from src.v7.common import (BOXES, BLOCK_PIX, PIX_PER_DEG, DATA_ROOT, YEAR, snap, crop_path, pixel_size_m, write_run_record)

PATCH = 40            # CCI pixels per side of the patch used for check 3
SUB = 5               # each 10 m pixel is split into SUB x SUB points to work out how much of it lies in each CCI pixel
SHIFTS = (-2, -1, 0, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def length(x):
    """Length of each embedding: square the 64 numbers, add, take the root. x: (..., 64)."""
    return np.sqrt((x.astype(np.float64) ** 2).sum(-1))


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def ridge_r2(x, y):
    """How much of the variation in y a straight-line model of the 64 numbers explains (fitted and scored on the same pixels)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y)
    return float(model.score(x, y))


def pick_patch(agb):
    """The PATCH x PATCH window of the block where biomass varies most (most mixed ground), away from the block edge."""
    best, where = -1.0, None
    fallback, fallback_where = -1.0, (10, 10)
    for r in range(10, agb.shape[0] - PATCH - 10, 5):
        for c in range(10, agb.shape[1] - PATCH - 10, 5):
            w = agb[r:r + PATCH, c:c + PATCH]
            spread = float(w.std())
            if spread > fallback:
                fallback, fallback_where = spread, (r, c)
            if 0.3 <= float((w >= 50).mean()) <= 0.7 and spread > best:      # roughly half forest, half open
                best, where = spread, (r, c)
    return where or fallback_where


def own_average(x10, valid10, t10, crs10, west, north, pix):
    """Average the 10 m embeddings onto the PATCH x PATCH CCI pixels with plain numpy.

    Each 10 m pixel is split into SUB x SUB points; each point is given to the CCI pixel
    it falls in. A 10 m pixel that straddles a CCI pixel edge therefore counts for the
    part inside, which is the same rule as Earth Engine's. Independent of Earth Engine's
    averaging: only the 10 m values come from it.
    """
    h, w, nb = x10.shape
    flat = x10.reshape(-1, nb).astype(np.float64)
    vflat = valid10.reshape(-1).astype(np.float64)
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    sums = np.zeros((PATCH * PATCH, nb)); wts = np.zeros(PATCH * PATCH); pts = np.zeros(PATCH * PATCH)
    for ky in range(SUB):
        for kx in range(SUB):
            xs = t10[2] + (jj + (kx + 0.5) / SUB) * t10[0]
            ys = t10[5] + (ii + (ky + 0.5) / SUB) * t10[4]
            lon, lat = warp_points(crs10, "EPSG:4326", xs.ravel(), ys.ravel())
            col = np.floor((np.asarray(lon) - west) / pix).astype(int)
            row = np.floor((north - np.asarray(lat)) / pix).astype(int)
            inside = (col >= 0) & (col < PATCH) & (row >= 0) & (row < PATCH)
            cell = (row * PATCH + col)[inside]
            v = vflat[inside]
            pts += np.bincount(cell, minlength=PATCH * PATCH)
            wts += np.bincount(cell, weights=v, minlength=PATCH * PATCH)
            for b in range(nb):
                sums[:, b] += np.bincount(cell, weights=flat[inside, b] * v, minlength=PATCH * PATCH)
    mean = sums / np.maximum(wts, 1e-12)[:, None]
    return mean.reshape(PATCH, PATCH, nb), (wts / np.maximum(pts, 1)).reshape(PATCH, PATCH), pts.reshape(PATCH, PATCH)


def spot_at_centres(x10, t10, crs10, west, north, pix):
    """The single 10 m embedding under the centre of each CCI pixel (what v6 used)."""
    cc, rr = np.meshgrid(np.arange(PATCH), np.arange(PATCH))
    lon = west + (cc + 0.5) * pix
    lat = north - (rr + 0.5) * pix
    x, y = warp_points("EPSG:4326", crs10, lon.ravel(), lat.ravel())
    j = np.floor((np.asarray(x) - t10[2]) / t10[0]).astype(int)
    i = np.floor((np.asarray(y) - t10[5]) / t10[4]).astype(int)
    return x10[i, j].reshape(PATCH, PATCH, -1)


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(path, agb, avg, lens, aspect, patch_rc, mine, ee_patch, spot, shift_r2, title, numbers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from sklearn.decomposition import PCA

    INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    blue = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    orange = LinearSegmentedColormap.from_list("orange", ["#fde6da", "#f9c3a8", "#f39a70", "#eb6834", "#c24f1f", "#8f3814", "#5e240c"])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 10.5))
    r0, c0 = patch_rc

    def frame(a):
        a.set_xticks([]); a.set_yticks([])
        a.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), PATCH, PATCH, fill=False, ec=INK, lw=1.2))
        for s in a.spines.values():
            s.set_color(AXIS)

    im = ax[0, 0].imshow(agb, cmap=blue, vmin=0, vmax=float(np.percentile(agb, 99)), aspect=aspect, interpolation="nearest")
    ax[0, 0].set_title("CCI biomass v7.0 (the target)", loc="left"); frame(ax[0, 0])
    cb = fig.colorbar(im, ax=ax[0, 0], fraction=0.04, pad=0.02); cb.set_label("Mg/ha", color=INK2); cb.outline.set_edgecolor(AXIS)

    im = ax[0, 1].imshow(lens, cmap=orange.reversed(), vmin=float(np.percentile(lens, 1)), vmax=1.0, aspect=aspect, interpolation="nearest")
    ax[0, 1].set_title("Length of the averaged embedding", loc="left"); frame(ax[0, 1])
    cb = fig.colorbar(im, ax=ax[0, 1], fraction=0.04, pad=0.02); cb.set_label("length (1 = even ground, lower = mixed)", color=INK2); cb.outline.set_edgecolor(AXIS)

    flat = avg.reshape(-1, avg.shape[-1])
    pcs = PCA(n_components=3, random_state=0).fit_transform((flat - flat.mean(0)) / flat.std(0))
    lo, hi = np.percentile(pcs, 2, axis=0), np.percentile(pcs, 98, axis=0)
    rgb = np.clip((pcs - lo) / (hi - lo), 0, 1).reshape(avg.shape[0], avg.shape[1], 3)
    ax[0, 2].imshow(rgb, aspect=aspect, interpolation="nearest")
    ax[0, 2].set_title("Averaged embeddings as a picture", loc="left"); frame(ax[0, 2])
    ax[0, 2].set_xlabel("colours only mean: same colour = similar ground", color=INK2)

    lim = float(np.percentile(np.abs(mine), 99.9)) * 1.05
    for a, other, name, key in ((ax[1, 0], ee_patch, "Earth Engine's average", "rms_ee_vs_own"), (ax[1, 1], spot, "one 10 m spot (the v6 way)", "rms_spot_vs_own")):
        hb = a.hexbin(mine.ravel(), other.ravel(), gridsize=70, extent=(-lim, lim, -lim, lim), bins="log", cmap=blue, mincnt=1, linewidths=0)
        a.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--")
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_xlabel("my own average of the 10 m embeddings (each band value)"); a.set_ylabel(name)
        a.set_title(f"{name[0].upper() + name[1:]} against my own average", loc="left")
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.text(0.04, 0.95, f"typical difference {numbers[key]:.4f}\n(a typical band value is 0.125)", transform=a.transAxes, va="top", color=INK,
               bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        cb = fig.colorbar(hb, ax=a, fraction=0.04, pad=0.02); cb.set_label("number of values", color=INK2); cb.outline.set_edgecolor(AXIS)

    a = ax[1, 2]
    grid = np.array([[shift_r2[(dy, dx)] for dx in SHIFTS] for dy in SHIFTS])
    im = a.imshow(grid, cmap=blue, interpolation="nearest")
    for i, dy in enumerate(SHIFTS):
        for j, dx in enumerate(SHIFTS):
            v = grid[i, j]
            a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="#ffffff" if v > (grid.min() + grid.max()) / 2 else INK,
                   fontweight="bold" if (dy, dx) == (0, 0) else "normal")
    a.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, ec=INK, lw=1.6))
    a.set_xticks(range(len(SHIFTS))); a.set_xticklabels([f"{d:+d}" if d else "0" for d in SHIFTS])
    a.set_yticks(range(len(SHIFTS))); a.set_yticklabels([f"{d:+d}" if d else "0" for d in SHIFTS])
    a.set_xlabel("CCI value taken this many pixels further east"); a.set_ylabel("CCI value taken this many pixels further south")
    a.set_title("Shift test: fit is best with no shift (centre)", loc="left")
    for s in a.spines.values():
        s.set_color(AXIS)
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.set_label("share of biomass variation explained (R²)", color=INK2); cb.outline.set_edgecolor(AXIS)

    fig.suptitle(title, x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.945, "Top: the whole block, 225 x 225 CCI pixels; the square marks the 40 x 40 pixel patch used in the bottom-left and bottom-middle panels.",
             color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--box", default="netherlands", choices=list(BOXES))
    p.add_argument("--block", default="r2_c3", help="r<row>_c<col>; row 0 is the northern row, col 0 the western column")
    p.add_argument("--results_root", default="results/v7/one_block_test")
    args = p.parse_args()
    br, bc = (int(v[1:]) for v in args.block.split("_"))
    tag = f"{args.box}_{args.block}_" + (f"job{os.getenv('SLURM_JOB_ID')}" if os.getenv("SLURM_JOB_ID") else time.strftime("%Y%m%d_%H%M%S"))
    res_dir = os.path.join(args.results_root, tag)
    data_dir = os.path.join(DATA_ROOT, "one_block_test", tag)
    for d in (res_dir, data_dir):
        if os.path.exists(d):
            raise SystemExit(f"{d} already exists; nothing is overwritten")

    # ---- the CCI side: the block, from the crop made by crop_cci.py ----
    s = snap(args.box)
    with rasterio.open(crop_path(args.box, "AGB")) as src:
        win = Window(bc * BLOCK_PIX, br * BLOCK_PIX, BLOCK_PIX, BLOCK_PIX)
        agb = src.read(1, window=win).astype(np.float32)
        cci_t = src.window_transform(win)
    pix = cci_t.a
    west, north = cci_t.c, cci_t.f
    file_grid = [pix, 0.0, float(s["lon_left"]), 0.0, -pix, float(s["lat_top"])]      # the grid of the whole CCI file
    block_grid = [pix, 0.0, west, 0.0, -pix, north]
    bounds = (west, north - BLOCK_PIX * pix, west + BLOCK_PIX * pix, north)
    ns_m, ew_m = pixel_size_m((bounds[1] + bounds[3]) / 2)
    print(f"block {args.box} {args.block}: {bounds[1]:.4f}-{bounds[3]:.4f} N, {bounds[0]:.4f}-{bounds[2]:.4f} E; CCI pixel {ns_m:.1f} x {ew_m:.1f} m")
    print(f"CCI biomass in the block: mean {agb.mean():.1f} Mg/ha, zero {100 * (agb == 0).mean():.1f} %, 50 or more {100 * (agb >= 50).mean():.1f} %")

    # ---- Earth Engine ----
    E.init()
    coll = E.collection(YEAR, bounds)
    ids = coll.aggregate_array("system:index").getInfo()
    stack, native = E.stack_10m(coll)
    ninfo = native.getInfo()
    crs10, nt = ninfo["crs"], ninfo["transform"]
    print(f"Earth Engine: {len(ids)} image(s) touch the block: {ids}; native grid {crs10}, pixel {nt[0]} m")

    t0 = time.time()
    arr, got_t, got_crs, names, nbytes = E.compute_pixels(E.averaged_on_cci_grid(stack, file_grid), "EPSG:4326", block_grid, BLOCK_PIX, BLOCK_PIX)
    secs_avg = time.time() - t0
    avg = np.moveaxis(arr[:64], 0, -1)            # (225, 225, 64)
    valid = arr[64]                               # share of each CCI pixel that had 10 m data
    print(f"averaged block delivered in {secs_avg:.1f} s, {nbytes / 1e6:.1f} MB, {arr.shape[0]} bands (64 embedding bands + valid)")

    t0 = time.time()
    arr_s, _, _, _, _ = E.compute_pixels(E.single_spot(coll), "EPSG:4326", block_grid, BLOCK_PIX, BLOCK_PIX)
    secs_spot = time.time() - t0
    spot_block = np.moveaxis(arr_s, 0, -1)

    # ---- check 1: grid ----
    grid_diff = max(abs(a - b) for a, b in zip((got_t.a, got_t.b, got_t.c, got_t.d, got_t.e, got_t.f), (cci_t.a, cci_t.b, cci_t.c, cci_t.d, cci_t.e, cci_t.f)))
    check_grid = bool(grid_diff < 1e-9 and avg.shape[:2] == (BLOCK_PIX, BLOCK_PIX) and got_crs.to_epsg() == 4326)

    # ---- check 2: lengths over the whole block ----
    lens = length(avg)
    lens_spot = length(spot_block)
    has_data = valid > 0

    # ---- check 3: my own average on a patch ----
    pr, pc = pick_patch(agb)
    p_west, p_north = west + pc * pix, north - pr * pix
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs10, p_west, p_north - PATCH * pix, p_west + PATCH * pix, p_north, densify_pts=21)
    # Earth Engine stores this grid bottom-up (rows count from south to north). The patch is
    # requested north-up on the same 10 m lattice: the same pixels, rows in the usual order.
    step = abs(nt[0])
    assert abs(abs(nt[4]) - step) < 1e-9 and nt[1] == 0 and nt[3] == 0, f"unexpected native grid {nt}"
    left = nt[2] + (np.floor((minx - nt[2]) / step) - 3) * step
    top = nt[5] + (np.ceil((maxy - nt[5]) / step) + 3) * step
    w10 = int(np.ceil((maxx - left) / step)) + 3
    h10 = int(np.ceil((top - miny) / step)) + 3
    t10 = [step, 0.0, float(left), 0.0, -step, float(top)]
    t0 = time.time()
    arr10, _, _, _, nbytes10 = E.compute_pixels(stack.unmask(0).toFloat(), crs10, t10, w10, h10)
    secs_10m = time.time() - t0
    x10 = np.moveaxis(arr10[:64], 0, -1)
    valid10 = arr10[64]
    print(f"10 m patch delivered in {secs_10m:.1f} s: {h10} x {w10} pixels, {nbytes10 / 1e6:.1f} MB; missing 10 m pixels {100 * (valid10 == 0).mean():.2f} %")

    mine, mine_valid, pts = own_average(x10, valid10, t10, crs10, p_west, p_north, pix)
    spot = spot_at_centres(x10, t10, crs10, p_west, p_north, pix)
    ee_patch = avg[pr:pr + PATCH, pc:pc + PATCH]
    agb_patch = agb[pr:pr + PATCH, pc:pc + PATCH]

    numbers = {
        "rms_ee_vs_own": rms(ee_patch - mine),
        "rms_spot_vs_own": rms(spot - mine),
        "rms_ee_vs_spot": rms(ee_patch - spot),
        "rms_ee_single_spot_request_vs_own": rms(spot_block[pr:pr + PATCH, pc:pc + PATCH] - mine),
        "typical_band_value": rms(mine),
        "length_ee_mean": float(length(ee_patch).mean()), "length_own_mean": float(length(mine).mean()),
        "length_mean_abs_diff": float(np.abs(length(ee_patch) - length(mine)).mean()),
        "length_10m_min": float(length(x10)[valid10 > 0].min()), "length_10m_max": float(length(x10)[valid10 > 0].max()),
        "points_per_cci_pixel_min": float(pts.min()), "points_per_cci_pixel_median": float(np.median(pts)),
    }
    check_length = bool(numbers["length_mean_abs_diff"] < 0.005 and 0.98 <= numbers["length_10m_min"] and numbers["length_10m_max"] <= 1.02)
    check_average = bool(numbers["rms_ee_vs_own"] < 0.005 and numbers["rms_ee_vs_own"] < 0.2 * numbers["rms_ee_vs_spot"])

    # ---- shift test and averaged-against-spot, whole block ----
    ok = has_data
    shift_r2 = {}
    m = max(abs(v) for v in SHIFTS)
    core = (slice(m, BLOCK_PIX - m), slice(m, BLOCK_PIX - m))
    xc = avg[core].reshape(-1, 64); okc = ok[core].reshape(-1)
    for dy in SHIFTS:
        for dx in SHIFTS:
            yc = agb[m + dy:BLOCK_PIX - m + dy, m + dx:BLOCK_PIX - m + dx].reshape(-1)
            shift_r2[(dy, dx)] = ridge_r2(xc[okc], yc[okc])
    best_shift = max(shift_r2, key=shift_r2.get)
    r2_avg = ridge_r2(avg[ok], agb[ok])
    r2_spot = ridge_r2(spot_block[ok], agb[ok])

    result = {
        "box": args.box, "block": args.block, "bounds_west_south_east_north": bounds,
        "cci_pixel_m": [ns_m, ew_m], "ee_images": ids, "native_crs": crs10, "native_pixel_m": nt[0], "native_transform": nt, "native_stored_bottom_up": bool(nt[4] > 0), "method": E.METHOD,
        "seconds": {"averaged_block": secs_avg, "single_spot_block": secs_spot, "10m_patch": secs_10m},
        "megabytes": {"averaged_block": nbytes / 1e6, "10m_patch": nbytes10 / 1e6},
        "check_1_grid": {"pass": check_grid, "largest_difference_deg": grid_diff, "delivered_transform": list(got_t)[:6], "cci_transform": list(cci_t)[:6],
                         "delivered_shape": list(avg.shape[:2]), "delivered_crs": str(got_crs)},
        "check_2_length": {"pass": check_length,
                           "block_averaged": {"mean": float(lens[has_data].mean()), "p01": float(np.percentile(lens[has_data], 1)), "p50": float(np.median(lens[has_data])),
                                              "max": float(lens[has_data].max()), "share_below_0.95": float((lens[has_data] < 0.95).mean())},
                           "block_single_spot": {"mean": float(lens_spot.mean()), "min": float(lens_spot.min()), "max": float(lens_spot.max())},
                           "patch": {k: numbers[k] for k in ("length_ee_mean", "length_own_mean", "length_mean_abs_diff", "length_10m_min", "length_10m_max")}},
        "check_3_average": {"pass": check_average, "patch_rows_cols": [pr, pc, PATCH], "patch_biomass_mean": float(agb_patch.mean()),
                            "patch_share_50_or_more": float((agb_patch >= 50).mean()),
                            **{k: numbers[k] for k in ("rms_ee_vs_own", "rms_spot_vs_own", "rms_ee_vs_spot", "rms_ee_single_spot_request_vs_own", "typical_band_value",
                                                       "points_per_cci_pixel_min", "points_per_cci_pixel_median")}},
        "missing": {"cci_pixels_with_no_10m_data": int((valid == 0).sum()), "cci_pixels_partly_missing": int(((valid > 0) & (valid < 0.999)).sum()),
                    "cci_pixels_total": int(valid.size)},
        "shift_test": {"best_shift_south_east": list(best_shift), "r2_no_shift": shift_r2[(0, 0)],
                       "r2_by_shift": {f"{dy:+d},{dx:+d}": v for (dy, dx), v in shift_r2.items()}},
        "averaged_against_single_spot": {"ridge_r2_averaged": r2_avg, "ridge_r2_single_spot": r2_spot,
                                         "note": "ridge fitted and scored on the same pixels of this one block; a quick indication, not a model result"},
        "all_three_checks_pass": bool(check_grid and check_length and check_average),
    }

    # ---- write ----
    os.makedirs(res_dir); os.makedirs(data_dir)
    with open(os.path.join(res_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    prof = dict(driver="GTiff", height=BLOCK_PIX, width=BLOCK_PIX, dtype="float32", crs="EPSG:4326", transform=got_t, compress="deflate")
    with rasterio.open(os.path.join(data_dir, "block_averaged.tif"), "w", count=65, **prof) as dst:
        dst.write(arr.astype(np.float32)); dst.descriptions = tuple(E.BANDS + ["valid"])
    with rasterio.open(os.path.join(data_dir, "block_single_spot.tif"), "w", count=64, **prof) as dst:
        dst.write(arr_s.astype(np.float32)); dst.descriptions = tuple(E.BANDS)
    np.savez_compressed(os.path.join(data_dir, "patch_10m.npz"), x10=x10, valid10=valid10, transform=np.array(t10), crs=crs10,
                        own_average=mine, patch_rows_cols=np.array([pr, pc, PATCH]))
    make_figure(os.path.join(res_dir, "one_block_test.png"), agb, avg, lens, ns_m / ew_m, (pr, pc), mine, ee_patch, spot, shift_r2,
                f"One-block test: {args.box} {args.block} ({bounds[1]:.2f}-{bounds[3]:.2f} N, {bounds[0]:.2f}-{bounds[2]:.2f} E), AlphaEarth {YEAR} averaged onto the CCI grid",
                numbers)
    write_run_record(res_dir, "one_block_test", settings={**vars(args), "patch_pixels": PATCH, "sub_points": SUB, "shifts": SHIFTS,
                                                          "method": E.METHOD, "max_pixels": E.MAX_PIXELS, "collection": E.COLLECTION, "year": YEAR,
                                                          "data_dir": data_dir}, results=result)

    print("\n================ one-block test ================")
    print(f"1 grid     {'PASS' if check_grid else 'FAIL'}   largest difference to the CCI grid {grid_diff:.2e} degree; {avg.shape[0]} x {avg.shape[1]}; {got_crs}")
    print(f"2 length   {'PASS' if check_length else 'FAIL'}   averaged block: mean {lens[has_data].mean():.3f}, median {np.median(lens[has_data]):.3f}, lowest 1 % below {np.percentile(lens[has_data], 1):.3f}, "
          f"max {lens[has_data].max():.3f}; single 10 m pixels {numbers['length_10m_min']:.3f}-{numbers['length_10m_max']:.3f}; "
          f"Earth Engine against my average: {numbers['length_ee_mean']:.4f} against {numbers['length_own_mean']:.4f} (mean difference {numbers['length_mean_abs_diff']:.4f})")
    print(f"3 average  {'PASS' if check_average else 'FAIL'}   typical difference Earth Engine - my average {numbers['rms_ee_vs_own']:.4f}; one 10 m spot - my average {numbers['rms_spot_vs_own']:.4f}; "
          f"typical band value {numbers['typical_band_value']:.3f}")
    print(f"missing    {result['missing']['cci_pixels_with_no_10m_data']} CCI pixels with no data, {result['missing']['cci_pixels_partly_missing']} partly missing, of {valid.size}")
    print(f"shift      best fit at shift {best_shift} (south, east); R2 with no shift {shift_r2[(0, 0)]:.3f}")
    print(f"ridge R2   averaged {r2_avg:.3f}   single spot {r2_spot:.3f}   (same pixels, this block only)")
    print(f"time       averaged block {secs_avg:.1f} s, {nbytes / 1e6:.1f} MB")
    print(f"results -> {res_dir}/   data -> {data_dir}/")


if __name__ == "__main__":
    main()
