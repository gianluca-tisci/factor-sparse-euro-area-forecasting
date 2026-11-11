"""Download one EA-MD-QD release from Zenodo into data/raw/ and record its provenance.

Run from the repo root:  uv run python -m src.data.download_ea_md_qd
The release is fixed below (D-08); a different release is a new decision.
"""

import hashlib
import json
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

RECORD = "23064892"  # Zenodo record of release 09.2026, "data as of 30-Sep-2026"
VERSION = "09.2026"
FILENAME = "EA-MD-QD-2026-09.zip"
OUT = Path("data/raw/ea_md_qd") / "2026-09"
PARTIAL = OUT.with_name(OUT.name + ".partial")  # renamed to OUT only after all checks pass


def get(url, **kwargs):
    """GET with three attempts: Zenodo returns occasional 5xx/timeouts."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            if attempt == 2 or (
                isinstance(error, requests.HTTPError) and error.response.status_code < 500
            ):
                raise
            time.sleep(10 * (attempt + 1))


def file_hash(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(
            f"{OUT} already exists; data/raw/ is read-only. Delete it by hand to re-download."
        )
    record = get(f"https://zenodo.org/api/records/{RECORD}").json()
    if record["metadata"]["version"] != VERSION:
        raise SystemExit(
            f"Record {RECORD} is version {record['metadata']['version']}, expected {VERSION}"
        )
    entry = next(f for f in record["files"] if f["key"] == FILENAME)
    algorithm, expected = entry["checksum"].split(":")

    # A .partial folder is only ever an interrupted earlier run of this script.
    shutil.rmtree(PARTIAL, ignore_errors=True)
    PARTIAL.mkdir(parents=True)
    archive = PARTIAL / FILENAME
    with get(entry["links"]["self"], stream=True) as r, archive.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if file_hash(archive, algorithm) != expected:
        raise SystemExit(f"{algorithm} mismatch for {archive}: Zenodo says {expected}")

    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise SystemExit(f"Unsafe path in archive: {name}")
        z.extractall(PARTIAL)

    files = sorted(p for p in PARTIAL.rglob("*") if p.is_file())
    manifest = {
        "source": f"https://zenodo.org/records/{RECORD}",
        "version": VERSION,
        "publication_date": record["metadata"]["publication_date"],
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "archive": FILENAME,
        f"archive_{algorithm}": expected,
        "sha256": {str(p.relative_to(PARTIAL)): file_hash(p, "sha256") for p in files},
    }
    (PARTIAL / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    PARTIAL.rename(OUT)
    print(f"EA-MD-QD {VERSION}: {len(files)} files in {OUT}, checksum verified")


if __name__ == "__main__":
    main()
