# Literature notes — embeddings → biomass (started 2026-09-24)

Working document. Each entry: what it is, the numbers that matter for us, and why it matters.
Numbers marked *(snippet)* come from a search-result summary, not from reading the paper; verify before citing.

## Our numbers, for reference

Model trained on 70 France tiles, evaluated on 15 held-out France tiles and 100 Sweden tiles. Target = ESA CCI AGB v6 (100 m), so these are **map-to-map** errors, not errors against field plots.

| | RMSE (Mg/ha) | R² | CCI's own SD (rms) | RMSE / SD |
|---|---|---|---|---|
| France test, SmallCNN | 26.3 | 0.82 | 35.8 | 0.73 |
| Sweden, SmallCNN | 27.0 | 0.65 | 37.5 | 0.72 |
| France test, ridge (linear) | 31.7 | 0.73 | | |

## 1. The embeddings

**Brown et al. 2025 — AlphaEarth Foundations** (Google DeepMind). arXiv:2507.22291. https://arxiv.org/abs/2507.22291
- 64-dim embedding per 10 m pixel, one per year 2017–2024, released in Earth Engine as `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- Trained on Sentinel-2, Landsat 8/9, Sentinel-1, PALSAR-2, ERA5-Land, GEDI, GRACE, GLO-30, NLCD, plus text. At inference only S2/S1/Landsat are needed.
- Evaluated on 15 tasks from 11 datasets (land cover, crops, trees, evapotranspiration, emissivity). Claim: ~23.9 % lower error than the next-best approach on average.
- **No biomass or canopy-height evaluation in the paper.** GEDI is only a training target. So "does it carry biomass information" is genuinely open from their side — that is our question.

**TESSERA** (Cambridge). arXiv:2506.20380. https://arxiv.org/abs/2506.20380
- 128-dim embedding per 10 m pixel from Sentinel-1 + Sentinel-2 time series. Open weights.
- *(snippet)* In a biomass regression benchmark: TESSERA RMSE 27.43 t/ha, AlphaEarth 29.59 t/ha, SkySense 30.78. Need to read which reference data and region that is.
- Relevant because GeoTessera is the planned alternative input; a same-pipeline comparison AEF vs TESSERA would be a clean contribution.

## 2. Embeddings → biomass, directly comparable work

**Sialelli, Scheibenreif, Wegner, Schindler 2026 — Above-ground Biomass Estimation with Geospatial Foundation Models** (ETH/UZH). arXiv:2608.04792, submitted to RSE. https://arxiv.org/abs/2608.04792
- Closest paper to ours. Benchmarks 11 foundation models + AlphaEarth + TESSERA on the AGBD dataset (~16 M GEDI L4A footprints, 2019–2020, 11 regions worldwide), all **frozen** with a small trained head.
- AlphaEarth with an MLP head: RMSE ~52 Mg/ha; with a small conv head ~51 Mg/ha — best of everything tested, beating the supervised baseline on raw Sentinel-2 + PALSAR (~54). Linear probe on AEF: ~60. TESSERA (subset only): ~56.
- Against an independent plot-based reference (AGBref, 10 km): their AEF model R² 0.70 / RMSE 34.3 / bias 4.4 vs ESA CCI R² 0.69 / RMSE 34.8 / bias 4.7 — "near parity" with CCI.
- Cross-region transfer: fine-tuning on a new region improves it substantially (Africa 40.8 → 33.7), South Asia is the hard case.
- **Why it matters:** independently confirms that frozen AEF + tiny head is competitive with CCI itself. Our numbers are lower (26 vs 52) because our target is the CCI map, not GEDI footprints; the comparison to make in a write-up is "AEF model ≈ CCI" (theirs) and "AEF model reproduces CCI to within CCI's own SD" (ours). Same story, two references.
- Also useful: they list AGBD/AGBref as reference datasets we could evaluate against later, to get a plot-based number.

**Spectral indices outperform AlphaEarth foundation embeddings for AGB estimation in a regenerating tropical Andean forest** (2026). ScienceDirect S2352938526002466 (not yet read; paywalled from here). https://www.sciencedirect.com/science/article/pii/S2352938526002466
- Counter-example: in one regenerating tropical forest with local plots, hand-made spectral indices beat AEF. Worth reading for *why* (small n? saturation? annual embedding vs seasonal signal?). Keeps us honest that AEF is not universally best.

## 3. The reference map and its uncertainty

**ESA CCI Biomass** project page. https://climate.esa.int/en/projects/biomass/ ; data: https://climate.esa.int/en/projects/biomass/data/
- 100 m global AGB maps for 2007, 2010, 2015–2022 (v6, Santoro & Cartus 2025); v7 adds 2023–2024. C-band Sentinel-1 + L-band ALOS PALSAR/PALSAR-2, informed by GEDI/ICESat forest height.
- Stated target: relative error < 20 % where AGB > 50 Mg/ha.
- Ships a per-pixel SD layer. On our tiles the SD rms is ~36–38 Mg/ha; our model error is ~0.73 of that.

**Araza et al. 2022 — A comprehensive framework for assessing the accuracy and uncertainty of global above-ground biomass maps.** RSE 268, 112917. https://doi.org/10.1016/j.rse.2022.112917 (not yet read; paywalled from here)
- The standard reference for how CCI-type maps are validated against plots and how much their reported uncertainty can be trusted. Needed for the "is the SD layer itself reliable" caveat.

**Avitabile & Camia 2018 — An assessment of forest biomass maps in Europe using harmonized national statistics and inventory plots.** For. Ecol. Manage. 409:489–498. https://pmc.ncbi.nlm.nih.gov/articles/PMC5806600/
- Four European biomass maps vs 26 countries' NFI statistics (~431k plots) and 22k plots at pixel level.
- Plot-level RMSE 78–80 Mg/ha (58–67 % relative); national-level bias −23 to −43 Mg/ha. All maps overestimate below ~100 Mg/ha and underestimate above.
- **Why it matters:** sets the scale. Map-vs-plot errors in Europe are ~80 Mg/ha; our map-vs-map error is ~26. So we are well inside the noise of the reference products, and the saturation pattern we see (under-prediction > 150 Mg/ha) is the same one every map has.

## 4. Saturation / SAR background (for the discussion section)

- SAR-based AGB saturates: C-band ~60 Mg/ha, L-band ~90–150 Mg/ha depending on forest; combined C+L pushes it up. European validation shows good agreement to ~250 and systematic underestimation from ~300 Mg/ha (search summaries; pick one primary source, e.g. the Europe-wide biomass density maps paper, ScienceDirect S2352340926000892, or Hoscilo et al. 2018 for Poland).
- Our scatter shows the same under-prediction above ~150 Mg/ha, inherited from the target.

## Reference range, one line for the deck

| Comparison | RMSE (Mg/ha) |
|---|---|
| European maps vs field plots (Avitabile & Camia 2018) | 78–80 |
| Frozen AlphaEarth + small head vs GEDI footprints, global (Sialelli 2026) | ~51–52 |
| Same, vs plot-based AGBref 10 km (Sialelli 2026); CCI on the same reference | 34.3; 34.8 |
| **Ours: frozen AlphaEarth + small CNN vs CCI map, France / Sweden** | **26.3 / 27.0** |
| CCI's own per-pixel SD on our tiles (rms) | 36–38 |

## To do

1. Read Sialelli 2026 properly (methods + Table 2); it is the paper to position against.
2. Get Araza 2022 and the Andean-forest paper through the institutional library.
3. Read the TESSERA paper's biomass benchmark section; note which reference data.
4. Pick one primary SAR-saturation reference.
5. Decide whether to add a plot-based evaluation (AGBD Lite or AGBref) so we have a non-map reference too.
