"""Download the MediaPipe Pose Landmarker model used by RateMyForm.

Source (official MediaPipe docs, Pose Landmarker "Models" table):
  https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
License: Apache License 2.0 (stated in the "MediaPipe BlazePose GHUM 3D"
model card linked from that page).

Usage:
    python scripts/download_model.py                 # full model (default)
    python scripts/download_model.py --variant lite  # smaller, faster, less accurate

The file is saved to models/pose_landmarker_<variant>.task and its SHA-256
is written to models/MODEL_PROVENANCE.md so the team can verify it later.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(URLS), default="full")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / f"pose_landmarker_{args.variant}.task"
    if target.exists() and not args.force:
        print(f"Already present: {target} ({target.stat().st_size:,} bytes). Use --force to re-download.")
        return 0

    url = URLS[args.variant]
    tmp = target.with_suffix(".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 16):
            out.write(chunk)

    # A .task file is a zip bundle; reject anything else (e.g., an HTML error page).
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        print("ERROR: downloaded file is not a valid .task bundle.", file=sys.stderr)
        return 1
    tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    size = target.stat().st_size
    print(f"Saved {target} ({size:,} bytes)\nSHA-256 {digest}")

    provenance = MODEL_DIR / "MODEL_PROVENANCE.md"
    line = f"| {target.name} | {size:,} | `{digest}` | {url} | {dt.date.today().isoformat()} |\n"
    if not provenance.exists():
        provenance.write_text(
            "# MediaPipe model provenance\n\n"
            "Models are downloaded by `scripts/download_model.py` from Google's official MediaPipe model\n"
            "storage. License: Apache License 2.0 (MediaPipe BlazePose GHUM 3D model card).\n"
            "Model files are not committed to git (see .gitignore).\n\n"
            "| File | Bytes | SHA-256 | Source URL | Downloaded |\n|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with provenance.open("a", encoding="utf-8") as f:
        f.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
