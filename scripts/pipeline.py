#Download PRISM.
#Download + aggregate NLDAS.
#Calculate county zonal statistics.
#Validate the county-day master.
#Calculate 1991-2020 baselines.
#Engineer daily drought features.
#Download weekly USDM DSCI.
#Calibrate nonnegative composite weights.


import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

ZONAL_FILE = OUTPUT_DIR / "zonal_stats_master_ee.csv"
BASELINE_FILE = OUTPUT_DIR / "baselines.csv"
FEATURES_FILE = OUTPUT_DIR / "features_master.csv"
WEIGHTS_FILE = OUTPUT_DIR / "phase4_calibrated_weights.csv"
MODEL_DATASET_FILE = OUTPUT_DIR / "model_dataset_2000_2025.csv"
DSCI_FILE = DATA_DIR / "usdm" / "usdm_county_dsci_2000_2025.csv"

PRISM_DIR = DATA_DIR / "prism_raw"
NLDAS_DAILY_DIR = DATA_DIR / "nldas_daily_v2"

SCRIPTS = {
    "prism": SCRIPT_DIR / "prism_downloader.py",
    "nldas": SCRIPT_DIR / "nldas_downloader.py",
    "zonal": SCRIPT_DIR / "zonal_stats.py",
    "validate": SCRIPT_DIR / "validate.py",
    "baselines": SCRIPT_DIR / "compute_baselines.py",
    "features": SCRIPT_DIR / "feature_engineering.py",
    "dsci": SCRIPT_DIR / "download_dsci.py",
    "calibrate": SCRIPT_DIR / "calibrate_weights.py",
}


def require_scripts():
    #checks if any of the pipeline scripts are missing
    missing = [
        str(path)
        for path in SCRIPTS.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The simplified pipeline is missing these scripts:\n"
            + "\n".join(missing)
        )


def run_script(label, script, arguments=None):
    #error messaging
    arguments = arguments or []

    command = [
        sys.executable,
        str(script),
        *[str(arg) for arg in arguments],
    ]

    try:
        subprocess.check_call(
            command,
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError:
        raise SystemExit(f"\nPipeline stopped. Stage failed: {label}")


def ensure_baseline_period():
    if not ZONAL_FILE.exists():
        raise FileNotFoundError(
            f"Cannot check baseline period; missing {ZONAL_FILE}"
        )

    dates = pd.read_csv(
        ZONAL_FILE,
        usecols=["date"],
        parse_dates=["date"],
    )["date"]

    needed_start = pd.Timestamp("1991-01-01")
    needed_end = pd.Timestamp("2020-12-31")

    actual_start = dates.min()
    actual_end = dates.max()

    if actual_start > needed_start or actual_end < needed_end:
        raise RuntimeError(
            "The zonal-statistics master does not cover the full "
            "1991-2020 baseline period.\n"
            f"Current range: {actual_start.date()} to {actual_end.date()}\n"
            "Run the missing years before computing baselines."
        )


def remove_if_requested(path, overwrite):
    #delete a generated output
    if path.exists() and overwrite:
        print(f"Removing old output: {path}")
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run the pipeline"
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=1991,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Use PRISM/NLDAS files that are already on disk.",
    )

    parser.add_argument(
        "--skip-dsci-download",
        action="store_true",
        help="Use an existing USDM DSCI CSV instead of downloading it again.",
    )

    parser.add_argument(
        "--overwrite-zonal",
        action="store_true",
        help="Delete the existing zonal_stats_master_ee.csv before extraction.",
    )

    parser.add_argument(
        "--reuse-zonal",
        action="store_true",
        help=(
            "Keep the existing zonal_stats_master_ee.csv and skip "
            "extraction (it is still validated)."
        ),
    )

    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Stop after zonal statistics and validation.",
    )

    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError("end-year must be >= start-year")

    if args.reuse_zonal and args.overwrite_zonal:
        raise ValueError("Choose --reuse-zonal or --overwrite-zonal, not both.")

    require_scripts()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DSCI_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"


    #downloads
    if not args.skip_downloads:
        run_script(
            "1/8 - Download PRISM",
            SCRIPTS["prism"],
            [
                start_date,
                end_date,
            ],
        )

        run_script(
            "2/8 - Download and aggregate NLDAS",
            SCRIPTS["nldas"],
            [
                args.start_year,
                args.end_year,
                "--workers",
                args.workers,
            ],
        )

 
    #county extraction
    if args.reuse_zonal:
        if not ZONAL_FILE.exists():
            raise FileNotFoundError(
                f"--reuse-zonal was used, but {ZONAL_FILE} does not exist."
            )

    else:
        remove_if_requested(
            ZONAL_FILE,
            args.overwrite_zonal,
        )

        if ZONAL_FILE.exists():
            raise FileExistsError(
                f"{ZONAL_FILE} already exists.\n"
                "Use --reuse-zonal to keep it, --overwrite-zonal for a clean "
                "rebuild, or run the individual zonal script with --append "
                "if you are adding new years."
            )

        run_script(
            "3/8 - Calculate county zonal statistics",
            SCRIPTS["zonal"],
            [
                args.start_year,
                args.end_year,
                "--prism-dir",
                PRISM_DIR,
                "--nldas-dir",
                NLDAS_DAILY_DIR,
                "--output",
                ZONAL_FILE,
            ],
        )

    run_script(
        "4/8 - Validate county-day master",
        SCRIPTS["validate"],
        [
            "--csv",
            ZONAL_FILE,
            "--start-year",
            args.start_year,
            "--end-year",
            args.end_year,
        ],
    )

    if args.data_only:
        print("\nDone (data-only run). Output saved to:")
        print(f"  {ZONAL_FILE}")
        return

    
    #baselines and daily features
    ensure_baseline_period()

    run_script(
        "5/8 - Calculate 1991-2020 baselines",
        SCRIPTS["baselines"],
    )

    run_script(
        "6/8 - Engineer daily drought features",
        SCRIPTS["features"],
    )


    #USDM target and calibration
    if args.skip_dsci_download:
        if not DSCI_FILE.exists():
            raise FileNotFoundError(
                f"--skip-dsci-download was used, but {DSCI_FILE} does not exist."
            )

    else:
        #USDM starts in 2000
        dsci_end_year = max(2000, args.end_year)

        run_script(
            "7/8 - Download USDM county DSCI",
            SCRIPTS["dsci"],
            [
                2000,
                dsci_end_year,
                "--output",
                DSCI_FILE,
            ],
        )

    run_script(
        "8/8 - Calibrate composite weights",
        SCRIPTS["calibrate"],
    )

   
    #rebuild composite using the newly learned weight CSV
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Calibration finished but did not create {WEIGHTS_FILE}"
        )


    print("\nPipeline done. Output saved to:")

    for path in [
        ZONAL_FILE,
        BASELINE_FILE,
        FEATURES_FILE,
        DSCI_FILE,
        WEIGHTS_FILE,
        MODEL_DATASET_FILE,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
