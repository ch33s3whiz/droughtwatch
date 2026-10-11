#for every county and calendar month, calculate the 1991-2020
# historical mean and standard deviation for each climate variable.


from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "output" / "zonal_stats_master_ee.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "baselines.csv"

BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"
NEAR_ZERO_STD = 1e-6

VARIABLES = [
    "tmax_mean",
    "tmin_mean",
    "soilm_surface_mean",
    "soilm_root_0_100_mean",
    "et_mean",
]

def check_columns(data):
    missing = [name for name in VARIABLES if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def calculate_baselines(data):
    data = data.copy()
    data["month"] = data["date"].dt.month
    rows = []

    for (county_fips, month), group in data.groupby(["county_fips", "month"]):
        row = {"county_fips": county_fips, "month": month}

        for variable in VARIABLES:
            row[f"{variable}_baseline_mean"] = group[variable].mean()
            row[f"{variable}_baseline_std"] = group[variable].std()

        rows.append(row)

    return pd.DataFrame(rows)


def clean_tiny_standard_deviations(baselines):
    for variable in VARIABLES:
        std_column = f"{variable}_baseline_std"

        tiny = (
            baselines[std_column].notna()
            & (baselines[std_column].abs() < NEAR_ZERO_STD)
        )

        count = int(tiny.sum())

        if count:
            print(
                f"{variable}: {count} near-zero standard deviations "
                "changed to NaN"
            )
            baselines.loc[tiny, std_column] = np.nan

    return baselines


def main():
    data = pd.read_csv(
        INPUT_FILE,
        dtype={"county_fips": str},
    )

    data["county_fips"] = data["county_fips"].str.zfill(5)
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    check_columns(data)

    baseline_data = data[
        (data["date"] >= BASELINE_START)
        & (data["date"] <= BASELINE_END)
    ].copy()

    if baseline_data.empty:
        raise ValueError("No rows were found in the 1991-2020 baseline period.")

    county_count = baseline_data["county_fips"].nunique()

    baselines = calculate_baselines(baseline_data)
    baselines = clean_tiny_standard_deviations(baselines)

    expected_rows = county_count * 12

    if len(baselines) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} baseline rows, "
            f"but created {len(baselines)}."
        )

    if baselines.duplicated(["county_fips", "month"]).any():
        raise ValueError("Duplicate county/month rows found.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Baseline rows: {len(baselines):,} "
        f"({county_count} counties x 12 months, "
        f"{baseline_data['date'].min().date()} to "
        f"{baseline_data['date'].max().date()})"
    )
    print("\nDone. Output saved to:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
