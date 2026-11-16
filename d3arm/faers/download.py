"""Download and unzip the FAERS quarterly ASCII files.

The FDA publishes one zip per quarter on the FAERS Quarterly Data Extract page:
https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

The file names are like faers_ascii_2023q1.zip. The case of the "q" has not
always been the same, so we try both. If a download keeps failing, download
the zip by hand from the page above and drop it in data/raw/faers/ - the
unzip step will pick it up.
"""
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://fis.fda.gov/content/Exports/"


def zip_name(quarter):
    year, q = quarter[:4], quarter[-1]
    return [f"faers_ascii_{year}q{q}.zip", f"faers_ascii_{year}Q{q}.zip"]


def download_quarter(quarter, out_dir: Path, retries=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in zip_name(quarter):
        target = out_dir / name.lower()
        if target.exists() and target.stat().st_size > 0:
            print(f"{quarter}: already have {target.name}")
            return target
    for name in zip_name(quarter):
        url = BASE_URL + name
        for attempt in range(1, retries + 1):
            try:
                print(f"{quarter}: downloading {url} (try {attempt})")
                tmp = out_dir / (name.lower() + ".part")
                urllib.request.urlretrieve(url, tmp)
                final = out_dir / name.lower()
                tmp.rename(final)
                return final
            except Exception as e:  # 404 for the wrong case, or a network hiccup
                print(f"  failed: {e}")
                time.sleep(2 * attempt)
                if "404" in str(e):
                    break
    print(f"{quarter}: could not download, get it manually from the FDA page")
    return None


def unzip_quarter(zip_path: Path, out_root: Path, quarter):
    """Extract only the tables we use (+ the deleted-cases list if present)."""
    keep = ("DEMO", "DRUG", "REAC", "INDI", "DELETE")
    out_dir = out_root / quarter
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            fname = Path(member).name
            if fname.upper().endswith(".TXT") and any(k in fname.upper() for k in keep):
                # flatten the folder structure (ascii/, ASCII/, Deleted/ ...)
                with z.open(member) as src, open(out_dir / fname.upper(), "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out_dir
