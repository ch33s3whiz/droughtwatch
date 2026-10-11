import argparse
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "prism_raw"

#PRISM variables 
VARIABLES = ["ppt", "tmax", "tmin"]
BASE_URL = "https://services.nacse.org/prism/data/get/us/4km"

#complete PRISM BIL dataset needs these three files
REQUIRED_EXTENSIONS = [".bil", ".hdr", ".prj"]


def each_day(start, end):
    #gives us every date from start through end
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def folder_is_complete(folder):
    #check whether BIL, HDR, and PRJ files already exist
    return all(any(folder.glob(f"*{ext}")) for ext in REQUIRED_EXTENSIONS)


def valid_zip(path):
    #check that a downloaded ZIP is real and not corrupted
    if not path.exists() or not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_file(url, final_path, retries=5):
    #download one file safely and return True if it worked
    temp_path = final_path.with_suffix(final_path.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            #sometimes a server sends an error page instead of a ZIP
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type or "json" in content_type:
                raise ValueError(f"Server returned {content_type} instead of a ZIP")

            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if not valid_zip(temp_path):
                raise ValueError("Downloaded file is not a valid ZIP")

            #rename after validation
            temp_path.replace(final_path)
            return True

        except Exception:
            temp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)

    return False


def download_prism_day(variable, day, output_dir):
    #download and extract one PRISM variable for one day
    day_text = day.strftime("%Y%m%d")
    folder = output_dir / variable / f"{variable}_{day_text}"

    if folder_is_complete(folder):
        return True

    folder.mkdir(parents=True, exist_ok=True)

    zip_path = folder / f"prism_{variable}_{day_text}.zip"
    url = f"{BASE_URL}/{variable}/{day_text}?format=bil"

    if not download_file(url, zip_path):
        print(f"FAIL {variable} {day_text}")
        return False

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(folder)
    except zipfile.BadZipFile:
        print(f"FAIL {variable} {day_text}: bad ZIP")
        return False

    if not folder_is_complete(folder):
        print(f"FAIL {variable} {day_text}: missing extracted files")
        return False

    #extracted BIL/HDR/PRJ files that project uses later
    zip_path.unlink(missing_ok=True)
    return True


def download_prism(start_text, end_text, output_dir):
    #ownload ppt, tmax, and tmin for every requested day
    start = date.fromisoformat(start_text)
    end = date.fromisoformat(end_text)

    if end < start:
        raise ValueError("End date must be on or after start date")

    failures = 0

    for variable in VARIABLES:
        for day in each_day(start, end):
            if not download_prism_day(variable, day, output_dir):
                failures += 1

            #pause so that we don't overwhelm the PRISM server
            time.sleep(2)

    if failures:
        print(f"\nFinished with {failures} failed file(s). Rerun to retry them.")
        raise SystemExit(1)

    print("Done. Output saved to:")
    print(f"  {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download daily PRISM data")
    parser.add_argument("start_date", help="Example: 2024-01-01")
    parser.add_argument("end_date", help="Example: 2024-12-31")
    args = parser.parse_args()

    download_prism(args.start_date, args.end_date, OUTPUT_DIR)


if __name__ == "__main__":
    main()
