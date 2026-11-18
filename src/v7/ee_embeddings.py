"""AlphaEarth embeddings from Earth Engine, averaged onto the CCI pixel grid.

Method (the same for every box; also written into the metadata table):
  Each CCI pixel gets the plain average of the 10 m embeddings that lie inside it.
  A 10 m pixel that is only partly inside counts for the part that is inside
  (Earth Engine's reduceResolution with a mean reducer weights each input pixel by
  the share of the output pixel it covers). The average is NOT rescaled to length 1.
  Missing 10 m pixels are left out of the average; the extra band 'valid' gives
  the share of the CCI pixel that had data (1 = all of it).

Why not the v6 way: by default Earth Engine picks one 10 m pixel per output pixel
(nearest neighbour). That pairs one 10 m spot with a 100 m CCI value.
"""
import os, time

import ee
import rasterio

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
MAX_PIXELS = 1024          # most 10 m pixels allowed per CCI pixel (about 50-125 are needed)
METHOD = ("mean of the 10 m AlphaEarth embeddings inside each CCI pixel, weighted by the share of the CCI pixel "
          "each 10 m pixel covers (ee.Image.reduceResolution, ee.Reducer.mean, maxPixels=1024, input in its native "
          "UTM 10 m grid), computed directly on the CCI grid (EPSG:4326, 1/1125 degree); not rescaled to length 1")


def init():
    ee.Initialize(project=os.getenv("EE_PROJECT", "esa-cci"))


def collection(year, bounds):
    """The annual embedding images that touch bounds = (west, south, east, north)."""
    geom = ee.Geometry.Rectangle(list(bounds), proj="EPSG:4326", geodesic=False)
    return ee.ImageCollection(COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(geom)


def stack_10m(coll):
    """64 embedding bands plus 'valid' (1 where the 10 m pixel has data, else 0), with the
    native 10 m grid of the first image as the grid Earth Engine works in. A mosaic has no
    grid of its own, and reduceResolution needs one."""
    native = coll.first().select(BANDS[0]).projection()
    mosaic = coll.mosaic().select(BANDS)
    valid = mosaic.select(BANDS[0]).mask().unmask(0).rename("valid")
    return mosaic.addBands(valid).setDefaultProjection(native), native


def averaged_on_cci_grid(stack, cci_transform):
    """cci_transform: the six grid numbers of the CCI file [pixel, 0, west, 0, -pixel, north]."""
    return (stack.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=MAX_PIXELS)
                 .reproject(crs="EPSG:4326", crsTransform=list(cci_transform))
                 .unmask(0).toFloat())


def single_spot(coll):
    """The v6 way, kept only for comparison: no averaging, Earth Engine picks one pixel."""
    return coll.mosaic().select(BANDS).unmask(0).toFloat()


def compute_pixels(image, crs, transform, width, height, tries=4):
    """Ask Earth Engine for exactly this grid. transform = [scaleX, 0, west, 0, scaleY, north].
    Returns (array (bands, height, width), transform, crs, band names) read back from the GeoTIFF
    it sends, so the grid that is reported is the grid that was delivered."""
    a, b, c, d, e, f = transform
    request = {
        "expression": image,
        "fileFormat": "GEO_TIFF",
        "grid": {"dimensions": {"width": int(width), "height": int(height)},
                 "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
                 "crsCode": crs},
    }
    last = None
    for attempt in range(tries):
        try:
            raw = ee.data.computePixels(request)
            break
        except Exception as err:        # Earth Engine is sometimes briefly busy; a real error repeats
            last = err
            print(f"  Earth Engine request failed (try {attempt + 1} of {tries}): {err}")
            time.sleep(5 * (attempt + 1))
    else:
        raise last
    with rasterio.MemoryFile(raw) as mem:
        with mem.open() as ds:
            return ds.read(), ds.transform, ds.crs, list(ds.descriptions), len(raw)
