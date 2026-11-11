# Data

Everything the code reads is in this folder, so a fresh clone needs no download.
Nothing in `raw/` or `external/` is edited by hand; the code checks each file against its `manifest.json` (source, retrieval time, SHA-256) before using it.

| Folder | What | Terms |
|---|---|---|
| `raw/ea_md_qd/2026-09/` | EA-MD-QD, release 09.2026 ("data as of 30-Sep-2026"), stored exactly as published on Zenodo | CC BY-NC 4.0 |
| `external/factor_sparse/` | the 24 external monthly series (commodity prices, business and consumer surveys, uncertainty, geopolitical-risk and supply-chain indices, HICP sub-indices) | each provider's terms, in `RIGHTS.md` |
| `metadata/` | `ea_md_qd_series.csv` (descriptions and units, from EA-MD-QD's data description, Table 2), `data_dictionary.csv` (one row per series: unit, transformation, sample, publication delay, role) and `variable_dictionary.csv` (short descriptions for the appendix series table) | ours |
| `interim/`, `processed/` | written by the code when it runs; not in git | |

## EA-MD-QD

The project uses only the euro-area file `EA-MD-QD-2026-09/EAdata.xlsx` (the series after the authors' seasonal adjustment).
The zip, the country files and the authors' routines are kept because the release is stored as published; the code does not read them.
To download the release again from Zenodo (record 23064892), with Zenodo's checksum checked:

```sh
uv run python -m src.data.download_ea_md_qd
```

It refuses to overwrite an existing folder: move `raw/ea_md_qd/2026-09/` away first.

Cite: Barigozzi, M. and Lissona, C. (2024), "EA-MD-QD: Large Euro Area and Euro Member Countries Datasets for Macroeconomic Research", Zenodo; and Barigozzi, M., Lissona, C. and Tonni, L., *Journal of Applied Econometrics* (2026), doi:10.1002/jae.70083.

## External series

`external/factor_sparse/manifest.json` gives, for each of the 24 series, its source, definition, frequency, transformation and assumed publication delay; `provenance/` keeps the original download metadata.
The CSVs hold levels; `src.data.external.load_external()` applies the log difference where the manifest asks for one.
Historical release dates and vintages are not reconstructed, so the exercise is pseudo-real-time on revised data.

The saved runs record a hash of `external/factor_sparse/README.md` and `RIGHTS.md` among their inputs, so both are kept byte for byte as they were when the forecasts were computed (including a note on internal review that no longer applies).

## Code that reads the data

- `src/data/ea_md_qd.py` loads the euro-area file and gives each series' publication delay and what was published by the end of any month.
- `src/data/external.py` loads the external series.
- `src/preprocessing/transform.py` applies the EA-MD-QD transformation codes.
- `src/data/dictionary.py` rebuilds `metadata/data_dictionary.csv`.
