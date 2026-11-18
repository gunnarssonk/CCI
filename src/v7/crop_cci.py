"""Step 2 of the v7 setup: check the CCI Biomass v7.0 files and crop each box.

    python -m src.v7.crop_cci

Reads the 2020 biomass (AGB) and uncertainty (AGB_SD) files straight from the
archive on JASMIN; nothing is downloaded. For each of the six boxes it
  - checks the file: lat/long grid (EPSG:4326), pixel 1/1125 degree, unsigned 16-bit;
  - cuts out the box, snapped to whole CCI pixels (see common.snap), unchanged values;
  - writes <DATA_ROOT>/cci/<box>_AGB_2020_fv7.0.tif and ..._AGB_SD_...tif;
  - writes two tables: boxes_v7.csv (one row per box) and blocks_cci_stats.csv
    (one row per block: how much of it is zero / low / high biomass).

A value of 0 is valid data (no biomass). The files carry no no-data value.
Existing output files are never overwritten: the box is skipped with a message.
"""
import os, csv, json

import numpy as np
import rasterio
from rasterio.windows import Window

from src.v7.common import (BOXES, BOX_PIX, BLOCK_PIX, PIX_PER_DEG, DATA_ROOT, YEAR, CCI_VERSION,
                           snap, cci_path, crop_path, blocks, pixel_size_m, write_run_record)

RANGES = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 400), (400, None)]   # Mg/ha, as in the evaluation


def check_and_read(path, s):
    """Header checks, then the box window. Returns (array, transform of the crop, crs, facts)."""
    with rasterio.open(path) as src:
        t = src.transform
        facts = {"crs": str(src.crs), "dtype": src.dtypes[0], "pixel_deg": t.a, "nodata": src.nodata,
                 "file_size_px": [src.width, src.height]}
        assert src.crs is not None and src.crs.to_epsg() == 4326, f"{path}: not EPSG:4326 ({src.crs})"
        assert src.dtypes[0] == "uint16", f"{path}: not unsigned 16-bit ({src.dtypes[0]})"
        assert abs(t.a - 1 / PIX_PER_DEG) < 1e-12 and abs(t.e + 1 / PIX_PER_DEG) < 1e-12, f"{path}: pixel is not 1/1125 degree ({t.a}, {t.e})"
        assert t.b == 0 and t.d == 0, f"{path}: grid is rotated"
        assert abs(t.c - s["lon_left"]) < 1e-9 and abs(t.f - s["lat_top"]) < 1e-9, f"{path}: file corner {t.c},{t.f} is not {s['lon_left']},{s['lat_top']}"
        assert s["col0"] + BOX_PIX <= src.width and s["row0"] + BOX_PIX <= src.height and min(s["col0"], s["row0"]) >= 0, f"{path}: box not inside the file"
        win = Window(s["col0"], s["row0"], BOX_PIX, BOX_PIX)
        arr = src.read(1, window=win)
        return arr, src.window_transform(win), src.crs, facts


def write_crop(path, arr, transform, crs):
    if os.path.exists(path):
        with rasterio.open(path) as old:
            same = np.array_equal(old.read(1), arr)
        print(f"  exists, not overwritten ({'identical' if same else 'DIFFERENT CONTENT: look at this'}): {path}")
        return
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="uint16", crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(arr, 1)
        dst.update_tags(source=f"ESA CCI Biomass {CCI_VERSION}, {YEAR}; crop of the archive file, values unchanged", units="Mg/ha")
    print(f"  written: {path}")


def main():
    out_dir = os.path.join(DATA_ROOT, "cci")
    os.makedirs(out_dir, exist_ok=True)
    box_rows, block_rows = [], []
    for box in BOXES:
        s = snap(box)
        print(f"\n{box} ({s['biome']}): file {s['tile']}, rows {s['row0']}..{s['row0'] + BOX_PIX}, cols {s['col0']}..{s['col0'] + BOX_PIX}")
        agb, transform, crs, facts = check_and_read(cci_path(s["tile"], "AGB"), s)
        sd, transform_sd, _, facts_sd = check_and_read(cci_path(s["tile"], "AGB_SD"), s)
        assert transform == transform_sd
        write_crop(crop_path(box, "AGB"), agb, transform, crs)
        write_crop(crop_path(box, "AGB_SD"), sd, transform, crs)

        mid_lat = (s["snapped"]["south"] + s["snapped"]["north"]) / 2
        ns_m, ew_m = pixel_size_m(mid_lat)
        row = {
            "box": box, "biome": s["biome"], "cci_file": s["tile"],
            **{f"requested_{k}": v for k, v in s["requested"].items()},
            **{f"snapped_{k}": round(v, 7) for k, v in s["snapped"].items()},
            "shift_north_m": round(s["shift_north_pixels"] * ns_m, 1), "shift_east_m": round(s["shift_east_pixels"] * ew_m, 1),
            "first_row_in_file": s["row0"], "first_col_in_file": s["col0"], "pixels": f"{BOX_PIX}x{BOX_PIX}",
            "pixel_north_south_m": round(ns_m, 1), "pixel_east_west_m": round(ew_m, 1),
            "crs": facts["crs"], "dtype": facts["dtype"], "pixel_deg": facts["pixel_deg"], "nodata_value_in_file": facts["nodata"],
            "agb_mean": round(float(agb.mean()), 1), "agb_max": int(agb.max()), "agb_pct_zero": round(100 * float((agb == 0).mean()), 1),
            "sd_mean": round(float(sd.mean()), 1),
        }
        for lo, hi in RANGES:
            sel = (agb >= lo) & (agb < hi) if hi else (agb >= lo)
            row[f"pct_{lo}_{hi if hi else 'up'}"] = round(100 * float(sel.mean()), 1)
        box_rows.append(row)
        print(f"  pixel {ns_m:.1f} m north-south x {ew_m:.1f} m east-west; moved {row['shift_north_m']} m north, {row['shift_east_m']} m east")
        print(f"  biomass mean {row['agb_mean']} Mg/ha, max {row['agb_max']}, zero {row['agb_pct_zero']} %; CCI SD mean {row['sd_mean']}")

        for name, r, c in blocks():
            b = agb[r * BLOCK_PIX:(r + 1) * BLOCK_PIX, c * BLOCK_PIX:(c + 1) * BLOCK_PIX]
            north = s["snapped"]["north"] - r * BLOCK_PIX / PIX_PER_DEG
            west = s["snapped"]["west"] + c * BLOCK_PIX / PIX_PER_DEG
            block_rows.append({
                "box": box, "block": name, "row": r, "col": c,
                "north": round(north, 6), "south": round(north - BLOCK_PIX / PIX_PER_DEG, 6),
                "west": round(west, 6), "east": round(west + BLOCK_PIX / PIX_PER_DEG, 6),
                "agb_mean": round(float(b.mean()), 1),
                "pct_zero": round(100 * float((b == 0).mean()), 1),
                "pct_1_50": round(100 * float(((b > 0) & (b < 50)).mean()), 1),
                "pct_50_150": round(100 * float(((b >= 50) & (b < 150)).mean()), 1),
                "pct_150_up": round(100 * float((b >= 150).mean()), 1),
            })

    for fname, rows in (("boxes_v7.csv", box_rows), ("blocks_cci_stats.csv", block_rows)):
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="") as f:      # tables are rebuilt from the crops on every run
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"table -> {path}")
    write_run_record(out_dir, "crop_cci", settings={"boxes": BOXES, "box_pixels": BOX_PIX, "block_pixels": BLOCK_PIX,
                                                    "cci_version": CCI_VERSION, "year": YEAR},
                     results={"boxes": box_rows})


if __name__ == "__main__":
    main()
