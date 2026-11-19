"""How far apart must two pixels be before their biomass stops being alike?

    python -m src.v7.alike_distance

Used to judge the block size: a block should be wider than the distance over which
the land stays alike (spatial autocorrelation), otherwise test blocks are too similar
to the train blocks next to them.

For every box, pairs of CCI pixels a given distance apart are compared, north-south
and east-west separately. The number reported is the correlation of their biomass:
1 = always the same, 0 = unrelated. Reads only the CCI crops made by crop_cci.py.
Writes results/v7/alike_distance/alike_distance.csv.
"""
import os, csv

import numpy as np
import rasterio

from src.v7.common import BOXES, BLOCK_PIX, snap, crop_path, pixel_size_m, write_run_record

KM = [0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]          # distances looked at
LEVELS = [0.5, 0.2]                                    # "alike" thresholds reported


def corr_at(a, lag, axis):
    """Correlation between every pixel and the pixel `lag` pixels further along `axis`."""
    if axis == 0:
        x, y = a[:-lag, :], a[lag:, :]
    else:
        x, y = a[:, :-lag], a[:, lag:]
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def first_below(dist, corr, level):
    """Distance (km) where the correlation first falls below `level`, by straight-line reading between the points."""
    d0, c0 = 0.0, 1.0
    for d, c in zip(dist, corr):
        if c < level:
            return d0 + (c0 - level) / (c0 - c) * (d - d0)
        d0, c0 = d, c
    return None


def main():
    rows = []
    for box in BOXES:
        s = snap(box)
        ns_m, ew_m = pixel_size_m((s["snapped"]["south"] + s["snapped"]["north"]) / 2)
        with rasterio.open(crop_path(box, "AGB")) as src:
            a = src.read(1).astype(np.float64)
        for axis, name, pix_m in ((0, "north-south", ns_m), (1, "east-west", ew_m)):
            lags = sorted({max(1, int(round(km * 1000 / pix_m))) for km in KM})
            lags = [l for l in lags if l < a.shape[axis] // 2]
            dist = [l * pix_m / 1000 for l in lags]
            corr = [corr_at(a, l, axis) for l in lags]
            row = {"box": box, "direction": name, "block_width_km": round(BLOCK_PIX * pix_m / 1000, 1)}
            for km in KM:
                l = max(1, int(round(km * 1000 / pix_m)))
                row[f"corr_at_{km}_km"] = round(corr[lags.index(l)], 3) if l in lags else ""
            for lev in LEVELS:
                d = first_below(dist, corr, lev)
                row[f"km_until_corr_below_{lev}"] = round(d, 1) if d is not None else f"more than {dist[-1]:.0f}"
            row["corr_at_block_width"] = round(corr_at(a, BLOCK_PIX, axis), 3)
            rows.append(row)

    out = "results/v7/alike_distance"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "alike_distance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'box':18s}{'direction':13s}{'block km':>9s}{'0.5 km':>8s}{'2 km':>7s}{'5 km':>7s}{'10 km':>7s}{'20 km':>7s}{'at block':>9s}{'below 0.5':>11s}{'below 0.2':>11s}")
    for r in rows:
        print(f"{r['box']:18s}{r['direction']:13s}{r['block_width_km']:>9.1f}{r['corr_at_0.5_km']:>8}{r['corr_at_2_km']:>7}{r['corr_at_5_km']:>7}"
              f"{r['corr_at_10_km']:>7}{r['corr_at_20_km']:>7}{r['corr_at_block_width']:>9}{str(r['km_until_corr_below_0.5']) + ' km':>11}{str(r['km_until_corr_below_0.2']) + ' km':>11}")
    write_run_record(out, "alike_distance", settings={"distances_km": KM, "levels": LEVELS, "block_pixels": BLOCK_PIX}, results={"rows": rows})


if __name__ == "__main__":
    main()
