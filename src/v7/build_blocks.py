"""Steps 3 and 4 of the v7 setup: build all blocks (embeddings + CCI biomass, on the CCI grid).

    python -m src.v7.build_blocks --dry_run true          # list what would be built; fetches no pixels
    sbatch --time=04:00:00 setup/jasmin/v7/cpu_job.sbatch src.v7.build_blocks
    python -m src.v7.build_blocks --boxes netherlands     # only some boxes

For every block (225 x 225 CCI pixels) of every box:
  input   AlphaEarth embeddings from Earth Engine, averaged onto the CCI grid
          (method in ee_embeddings.py; checked by one_block_test.py)
  target  CCI biomass v7.0 and its SD, cut from the crops made by crop_cci.py

Output, one folder per box, in the layout train.py / evaluate.py already read:
    <DATA_ROOT>/blocks/<box>/ae_embeddings/<box>_r<row>_c<col>_ae.npy   (225, 225, 64) float32
    <DATA_ROOT>/blocks/<box>/targets/<...>_y.npy                        (225, 225, 1)  float32, Mg/ha
    <DATA_ROOT>/blocks/<box>/targets_sd/<...>_sd.npy                    (225, 225, 1)  float32, Mg/ha
    <DATA_ROOT>/blocks/<box>/valid/<...>_valid.npy                      (225, 225)     share of the pixel that had 10 m data
    <DATA_ROOT>/blocks/<box>/manifest.json
    <DATA_ROOT>/blocks/blocks_metadata.csv        one row per block (the metadata table)
    <DATA_ROOT>/blocks/embeddings_metadata.json   properties of the embedding dataset and the method

Missing data: a CCI pixel is left out when any 10 m pixel inside it has no embedding
(valid below 0.999). Its 64 values are then set to the no-data marker that
src/dataset/dataset.py already recognises, so training and scoring skip it. CCI itself
has no missing values in the files (0 is valid data: no biomass).

Resumable: a block whose files exist is skipped; nothing is overwritten. Files are
written under a temporary name first, so a stopped job leaves no half-written file.
"""
import os, csv, json, time, argparse

import numpy as np
import rasterio
from rasterio.windows import Window

from src.v7 import ee_embeddings as E
from src.v7.common import (BOXES, BOX_PIX, BLOCK_PIX, DATA_ROOT, YEAR, CCI_VERSION, snap, crop_path, blocks, pixel_size_m, write_run_record)

NODATA_MARKER = np.float32(np.finfo(np.float32).min)     # -3.4e38; dataset.py treats anything below -1e30 as missing
MIN_VALID = 0.999


def str2bool(v):
    return str(v).lower() in ("true", "t", "yes", "y", "1")


def utm_zone_name(lat, lon):
    """The UTM zone a point lies in, as Earth Engine writes it in the UTM_ZONE property, e.g. '31N'."""
    return f"{int((lon + 180) // 6) + 1}{'N' if lat >= 0 else 'S'}"


def save_new(path, array):
    """np.save, but never over an existing file and never leaving a half-written one."""
    if os.path.exists(path):
        return False
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)
    return True


def images_for(bounds):
    """The embedding images to use for this ground: those of the UTM zone the centre lies in.
    Images of the neighbouring zone can overlap the edge; they are the same data on another
    grid, and mixing two grids inside one average is avoided. Falls back to all images if
    the zone filter finds none."""
    import ee
    west, south, east, north = bounds
    zone = utm_zone_name((south + north) / 2, (west + east) / 2)
    coll_all = E.collection(YEAR, bounds)
    coll = coll_all.filter(ee.Filter.eq("UTM_ZONE", zone))
    ids = coll.aggregate_array("system:index").getInfo()
    if ids:
        return coll, ids, zone, True
    return coll_all, coll_all.aggregate_array("system:index").getInfo(), zone, False


def build_block(box, name, r, c, s, agb_src, sd_src, out_dir, meta_log):
    full = f"{box}_{name}"
    paths = {
        "ae": os.path.join(out_dir, "ae_embeddings", f"{full}_ae.npy"),
        "y": os.path.join(out_dir, "targets", f"{full}_y.npy"),
        "sd": os.path.join(out_dir, "targets_sd", f"{full}_sd.npy"),
        "valid": os.path.join(out_dir, "valid", f"{full}_valid.npy"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        return "exists"

    win = Window(c * BLOCK_PIX, r * BLOCK_PIX, BLOCK_PIX, BLOCK_PIX)
    agb = agb_src.read(1, window=win).astype(np.float32)
    sd = sd_src.read(1, window=win).astype(np.float32)
    t = agb_src.window_transform(win)
    pix, west, north = t.a, t.c, t.f
    bounds = (west, north - BLOCK_PIX * pix, west + BLOCK_PIX * pix, north)
    file_grid = [pix, 0.0, float(s["lon_left"]), 0.0, -pix, float(s["lat_top"])]
    block_grid = [pix, 0.0, west, 0.0, -pix, north]

    coll, ids, zone, zone_filter = images_for(bounds)
    if not ids:
        raise RuntimeError(f"{full}: Earth Engine has no {YEAR} embedding image here")
    stack, native = E.stack_10m(coll)
    ninfo = native.getInfo()
    t0 = time.time()
    arr, got_t, got_crs, _, nbytes = E.compute_pixels(E.averaged_on_cci_grid(stack, file_grid), "EPSG:4326", block_grid, BLOCK_PIX, BLOCK_PIX)
    secs = time.time() - t0

    grid_diff = max(abs(a - b) for a, b in zip((got_t.a, got_t.b, got_t.c, got_t.d, got_t.e, got_t.f), (t.a, t.b, t.c, t.d, t.e, t.f)))
    if grid_diff > 1e-9 or arr.shape != (65, BLOCK_PIX, BLOCK_PIX) or got_crs.to_epsg() != 4326:
        raise RuntimeError(f"{full}: delivered grid differs from the CCI grid (difference {grid_diff}, shape {arr.shape}, {got_crs})")

    x = np.moveaxis(arr[:64], 0, -1).astype(np.float32)
    valid = arr[64].astype(np.float32)
    left_out = valid < MIN_VALID
    keep = ~left_out
    lens = np.sqrt((x[keep].astype(np.float64) ** 2).sum(-1)) if keep.any() else np.array([np.nan])
    vmin, vmax = (float(x[keep].min()), float(x[keep].max())) if keep.any() else (float("nan"), float("nan"))
    x[left_out] = NODATA_MARKER

    save_new(paths["y"], agb[..., None])
    save_new(paths["sd"], sd[..., None])
    save_new(paths["valid"], valid)
    save_new(paths["ae"], x)            # last: its presence marks the block as complete

    ns_m, ew_m = pixel_size_m((bounds[1] + bounds[3]) / 2)
    row = {
        "box": box, "block": name, "biome": s["biome"], "row": r, "col": c,
        "south": round(bounds[1], 7), "north": round(bounds[3], 7), "west": round(bounds[0], 7), "east": round(bounds[2], 7),
        "cci_pixel_north_south_m": round(ns_m, 1), "cci_pixel_east_west_m": round(ew_m, 1),
        "ten_m_pixels_per_cci_pixel": round(ns_m * ew_m / (abs(ninfo["transform"][0]) ** 2), 1),
        "ee_images": ";".join(ids), "utm_zone": zone, "zone_filter_used": zone_filter, "native_crs": ninfo["crs"],
        "native_pixel_m": abs(ninfo["transform"][0]),
        "bands": 64, "stored_dtype": "float32", "value_min": round(vmin, 4), "value_max": round(vmax, 4),
        "length_mean": round(float(np.nanmean(lens)), 4), "length_p01": round(float(np.nanpercentile(lens, 1)), 4),
        "pixels": int(valid.size), "pixels_left_out_missing_embedding": int(left_out.sum()), "pixels_with_no_embedding_at_all": int((valid == 0).sum()),
        "agb_mean": round(float(agb.mean()), 1), "agb_pct_zero": round(100 * float((agb == 0).mean()), 1), "sd_mean": round(float(sd.mean()), 1),
        "seconds": round(secs, 1), "megabytes": round(nbytes / 1e6, 1), "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_log, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def write_tables(root, image_props):
    """The metadata table (one row per block) and the description of the embedding dataset."""
    log = os.path.join(root, "blocks_metadata.jsonl")
    rows = {}
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                rows[(r["box"], r["block"])] = r
    order = {b: i for i, b in enumerate(BOXES)}
    rows = sorted(rows.values(), key=lambda r: (order[r["box"]], r["row"], r["col"]))
    if rows:
        with open(os.path.join(root, "blocks_metadata.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    meta = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {
            "dataset": E.COLLECTION, "source": "Google Earth Engine", "year": YEAR,
            "bands": "A00..A63 (64 numbers per pixel, no unit)", "value_range_on_page": [-1, 1],
            "original_pixel_m": 10, "original_grid": "local UTM zone (given per block in blocks_metadata.csv); stored bottom-up in Earth Engine",
            "original_length": "every 10 m embedding has length 1",
            "image_properties_example": image_props,
            "how_it_was_put_on_the_cci_grid": E.METHOD,
            "stored_as": "float32, (225, 225, 64) per block; length below 1 where the ground inside a CCI pixel is mixed",
            "missing": f"a CCI pixel is left out when less than {MIN_VALID} of it had 10 m data; its 64 values are {float(NODATA_MARKER):.7e}; "
                       "valid/<name>_valid.npy keeps the share that had data",
        },
        "target": {
            "dataset": f"ESA CCI Biomass {CCI_VERSION}, above-ground biomass (AGB) and its standard deviation (AGB_SD)", "year": YEAR,
            "source": "CEDA archive on JASMIN, /neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/2020", "unit": "Mg/ha",
            "grid": "EPSG:4326, pixel 1/1125 degree (about 99 m north-south; east-west 44-99 m depending on latitude)",
            "missing": "no no-data value in the files; 0 is valid data (no biomass)", "changes": "none: values are copied, the box is only cut out",
        },
        "boxes": {b: snap(b) for b in BOXES},
        "block_pixels": BLOCK_PIX, "blocks_per_box": (BOX_PIX // BLOCK_PIX) ** 2, "blocks_built": len(rows),
    }
    with open(os.path.join(root, "embeddings_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", default=",".join(BOXES), help="comma-separated box names; default all six")
    p.add_argument("--dry_run", type=str2bool, default=False, help="true: list what would be built, fetch no pixels")
    p.add_argument("--limit", type=int, default=None, help="stop after this many newly built blocks (for a test)")
    args = p.parse_args()
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    for b in boxes:
        if b not in BOXES:
            raise SystemExit(f"unknown box {b!r}; choose from {', '.join(BOXES)}")

    root = os.path.join(DATA_ROOT, "blocks")
    meta_log = os.path.join(root, "blocks_metadata.jsonl")
    E.init()
    built = skipped = 0
    failed, image_props = [], None
    t_start = time.time()
    for box in boxes:
        s = snap(box)
        out_dir = os.path.join(root, box)
        b = s["snapped"]
        coll, ids, zone, zone_filter = images_for((b["west"], b["south"], b["east"], b["north"]))
        if image_props is None and ids:
            props = coll.first().getInfo()["properties"]
            image_props = {k: v for k, v in props.items() if k != "system:footprint"}
        done = sum(os.path.exists(os.path.join(out_dir, "ae_embeddings", f"{box}_{name}_ae.npy")) for name, _, _ in blocks())
        print(f"\n{box}: UTM zone {zone}{'' if zone_filter else ' (zone filter found nothing, using all images)'}, {len(ids)} image(s) {ids}; {done} of {len(blocks())} blocks already built")
        if args.dry_run:
            continue
        for sub in ("ae_embeddings", "targets", "targets_sd", "valid"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        with rasterio.open(crop_path(box, "AGB")) as agb_src, rasterio.open(crop_path(box, "AGB_SD")) as sd_src:
            for name, r, c in blocks():
                if args.limit is not None and built >= args.limit:
                    break
                try:
                    res = build_block(box, name, r, c, s, agb_src, sd_src, out_dir, meta_log)
                except Exception as err:
                    failed.append((box, name, str(err)))
                    print(f"  FAILED {box} {name}: {err}")
                    continue
                if res == "exists":
                    skipped += 1
                    continue
                built += 1
                print(f"  {box} {name}: {res['seconds']} s, {res['megabytes']} MB, length mean {res['length_mean']}, "
                      f"left out {res['pixels_left_out_missing_embedding']} px, biomass mean {res['agb_mean']}")
        manifest = os.path.join(out_dir, "manifest.json")
        if not os.path.exists(manifest):
            with open(manifest, "w") as f:
                json.dump({"setup": "v7", "box": s, "year": YEAR, "cci_version": CCI_VERSION, "block_pixels": BLOCK_PIX,
                           "input": E.COLLECTION, "method": E.METHOD, "missing_rule": f"left out when valid < {MIN_VALID}",
                           "target": {"source": "ceda-archive", "version": CCI_VERSION, "crop": crop_path(box, "AGB")}}, f, indent=2, default=str)

    if args.dry_run:
        print("\ndry run: nothing fetched, nothing written")
        return
    rows = write_tables(root, image_props)
    print(f"\nbuilt {built}, already there {skipped}, failed {len(failed)}; {len(rows)} blocks in the metadata table; {(time.time() - t_start) / 60:.1f} minutes")
    for f in failed:
        print("  failed:", f)
    write_run_record(root, "build_blocks", settings={**vars(args), "block_pixels": BLOCK_PIX, "min_valid": MIN_VALID, "method": E.METHOD,
                                                     "max_pixels": E.MAX_PIXELS, "collection": E.COLLECTION, "year": YEAR, "cci_version": CCI_VERSION},
                     results={"built": built, "already_there": skipped, "failed": failed, "blocks_in_table": len(rows)})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
