# External snapshot for the proposed factor–sparse experiment

All 24 CSVs are byte-for-byte copies from source commit
`7f5d2100e155c02eb2cf8ab0b0cb2741c2d0566d`, branch `paper/clean-release`
of `PietroMenci/FarmPredict-Geopolitical-Shocks`.
`manifest.json` records original paths, hashes, definitions, frequencies,
transformations and provisional publication lags. Original metadata and download
provenance are retained under `provenance/`.

These are imported source observations, so they are retained here rather than
in the ignored, regenerable `data/processed/` directory. Existing EA-MD-QD files
are untouched. `src.data.external.load_external()` needs only this snapshot.

The source calls these CSVs processed, but they contain **levels**. The loader
applies the specified log difference once; six M6 indicators remain in levels.
M7 metadata's 312 observations refers to its 2000–2025 audit window, while each
file retains the full source history. The individual CSVs, rather than an old
model's combined/transformed panel, are authoritative.

Historical release dates and vintages are not reconstructed. GSCPI's early
history predates its publication; seasonal adjustments and histories are revised.
This supports only a retrospective pseudo-real-time exercise.

Provider attribution and reuse conditions are recorded in `RIGHTS.md`.
No dataset has been omitted. The European EPU provider page explicitly uses
CC BY 4.0. All files remain uncommitted pending the required human review.
