#Checks code for:
#Correct schema and corrected root-zone names.
#Exactly 158 counties and one row per county per date.
#Complete yearly row counts.
#No duplicate county/date rows.
#Reasonable physical value ranges.
#Missing-value rates.
#NLDAS coverage 

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "output" / "zonal_stats_master_ee.csv"

EXPECTED_COUNTIES = 158
EXPECTED_STATES = {"MD", "VA", "DC"}

REQUIRED_COLUMNS = [
    "county_fips",
    "county_name",
    "state",
    "date",
    "precip_mean",
    "precip_std",
    "tmax_mean",
    "tmax_std",
    "tmin_mean",
    "soilm_surface_mean",
    "soilm_surface_std",
    "soilm_root_0_100_mean",
    "soilm_root_0_100_std",
    "et_mean",
    "nldas_n_valid_pixels",
    "nldas_n_total_pixels",
    "prism_variables_found",
]

PRISM_COLUMNS = [
    "precip_mean",
    "precip_std",
    "tmax_mean",
    "tmax_std",
    "tmin_mean",
]

NLDAS_COLUMNS = [
    "soilm_surface_mean",
    "soilm_surface_std",
    "soilm_root_0_100_mean",
    "soilm_root_0_100_std",
    "et_mean",
]


#these are intended to catch unit mistakes, fill values, and check for broken extraction
VALUE_RANGES = {
    "precip_mean": (0.0, 500.0),
    "precip_std": (0.0, 300.0),
    "tmax_mean": (-50.0, 55.0),
    "tmax_std": (0.0, 25.0),
    "tmin_mean": (-60.0, 45.0),
    "soilm_surface_mean": (0.0, 500.0),
    "soilm_surface_std": (0.0, 250.0),
    "soilm_root_0_100_mean": (0.0, 1500.0),
    "soilm_root_0_100_std": (0.0, 750.0),
    "et_mean": (-5.0, 20.0),
}


def load_data(csv_path):
    #load the zonal-statistics master with FIPS/date types
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing zonal-statistics file: {csv_path}")

    data = pd.read_csv(
        csv_path,
        dtype={"county_fips": str},
    )

    data["county_fips"] = data["county_fips"].str.zfill(5)
    data["date"] = pd.to_datetime(
        data["date"],
        format="%Y-%m-%d",
        errors="raise",
    )

    return data


def check_schema(data):
    #require the correct schema
    print("\n[1] Schema")

    problems = []

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        problems.append(
            f"required columns are missing: {missing}"
        )

    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}")
        return False

    print("  PASS: corrected zonal-statistics schema is present")
    return True


def check_identifiers(data):
    #check FIPS, states, county count, and duplicate county/date rows
    print("\n[2] County identifiers")

    ok = True

    valid_fips = data["county_fips"].str.fullmatch(r"\d{5}", na=False)
    if not valid_fips.all():
        print(
            f"  FAIL: {(~valid_fips).sum():,} rows have invalid five-digit FIPS"
        )
        ok = False
    else:
        print("  PASS: all FIPS values are five digits")

    states = set(data["state"].dropna().astype(str).unique())

    if states != EXPECTED_STATES:
        print(
            f"  FAIL: state values are {sorted(states)}; "
            f"expected exactly {sorted(EXPECTED_STATES)}"
        )
        ok = False
    else:
        print(f"  PASS: state values are {sorted(states)}")

    expected_state_from_fips = {
        "11": "DC",
        "24": "MD",
        "51": "VA",
    }

    fips_state = data["county_fips"].str[:2].map(expected_state_from_fips)
    bad_state_match = fips_state.isna() | (fips_state != data["state"])

    if bad_state_match.any():
        print(
            f"  FAIL: {int(bad_state_match.sum()):,} rows have a "
            "FIPS prefix that does not match the state field"
        )
        ok = False
    else:
        print("  PASS: FIPS prefixes match MD/VA/DC state labels")

    county_count = data["county_fips"].nunique()
    if county_count != EXPECTED_COUNTIES:
        print(
            f"  FAIL: found {county_count} counties; "
            f"expected {EXPECTED_COUNTIES}"
        )
        ok = False
    else:
        print(f"  PASS: {county_count} unique counties")

    duplicates = data.duplicated(
        ["county_fips", "date"]
    ).sum()

    if duplicates:
        print(
            f"  FAIL: {duplicates:,} duplicate county/date rows"
        )
        ok = False
    else:
        print("  PASS: no duplicate county/date rows")

    return ok


def check_dates_and_rows(data, start_year=None, end_year=None):
    #check that every included date has exactly 158 counties and every requested
    #year has the expected total row count
    
    print("\n[3] Date and row completeness")

    ok = True

    counts_by_date = data.groupby("date")["county_fips"].nunique()
    bad_dates = counts_by_date[counts_by_date != EXPECTED_COUNTIES]

    if not bad_dates.empty:
        sample = {
            str(date.date()): int(count)
            for date, count in bad_dates.head(5).items()
        }
        print(
            f"  FAIL: {len(bad_dates):,} dates do not contain "
            f"{EXPECTED_COUNTIES} counties. Sample: {sample}"
        )
        ok = False
    else:
        print(
            f"  PASS: every stored date contains "
            f"{EXPECTED_COUNTIES} counties"
        )

    if start_year is None:
        start_year = int(data["date"].dt.year.min())

    if end_year is None:
        end_year = int(data["date"].dt.year.max())

    for year in range(start_year, end_year + 1):
        year_rows = data[data["date"].dt.year == year]

        days = 366 if calendar.isleap(year) else 365
        expected_rows = EXPECTED_COUNTIES * days
        actual_rows = len(year_rows)

        if actual_rows != expected_rows:
            print(
                f"  FAIL: {year} has {actual_rows:,} rows; "
                f"expected {expected_rows:,}"
            )
            ok = False
        else:
            print(
                f"  PASS: {year} has {actual_rows:,} rows"
            )

        expected_dates = pd.date_range(
            f"{year}-01-01",
            f"{year}-12-31",
            freq="D",
        )

        actual_dates = pd.DatetimeIndex(
            year_rows["date"].drop_duplicates().sort_values()
        )

        missing_dates = expected_dates.difference(actual_dates)

        if len(missing_dates):
            print(
                f"        missing dates: {len(missing_dates)} "
                f"(first: {missing_dates[0].date()})"
            )

    return ok


def check_prism_completeness(data):
    #require all three PRISM variables for every written day
    print("\n[4] PRISM completeness")

    expected = {"ppt", "tmax", "tmin"}

    def complete(value):
        found = {
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        }
        return found == expected

    complete_rows = data["prism_variables_found"].map(complete)
    bad_count = int((~complete_rows).sum())

    if bad_count:
        print(
            f"  FAIL: {bad_count:,} rows do not report all "
            "ppt, tmax, and tmin inputs"
        )
        return False

    print("  PASS: all rows report ppt, tmax, and tmin")
    return True


def check_numeric_values(data):
    #check infinities and physical ranges
    print("\n[5] Numeric sanity")

    ok = True

    numeric = data.select_dtypes(include=[np.number])

    if np.isinf(numeric.to_numpy()).any():
        print("  FAIL: one or more numeric columns contain +inf or -inf")
        ok = False
    else:
        print("  PASS: no infinite numeric values")

    for column, (minimum, maximum) in VALUE_RANGES.items():
        if column not in data.columns:
            continue

        values = data[column].dropna()

        if values.empty:
            print(f"  FAIL: {column} has no non-null values")
            ok = False
            continue

        bad = values[
            (values < minimum)
            | (values > maximum)
        ]

        if len(bad):
            print(
                f"  FAIL: {column} has {len(bad):,} values outside "
                f"[{minimum}, {maximum}]. "
                f"Observed {values.min():.3f} to {values.max():.3f}"
            )
            ok = False
        else:
            print(
                f"  PASS: {column}: "
                f"{values.min():.3f} to {values.max():.3f}"
            )

    return ok


def check_missing_values(data):
    #check for missing data
    print("\n[6] Missing values")

    ok = True

    for column in PRISM_COLUMNS:
        percent = float(data[column].isna().mean() * 100)

        if percent > 0.1:
            print(
                f"  FAIL: {column} is {percent:.2f}% missing "
                "(expected near 0%)"
            )
            ok = False
        else:
            print(
                f"  PASS: {column} is {percent:.2f}% missing"
            )

    for column in NLDAS_COLUMNS:
        percent = float(data[column].isna().mean() * 100)

        if percent > 2.0:
            print(
                f"  FAIL: {column} is {percent:.2f}% missing "
                "(unexpected with coverage-weighted extraction)"
            )
            ok = False
        else:
            print(
                f"  PASS: {column} is {percent:.2f}% missing"
            )

    return ok


def check_nldas_coverage(data):
    #check coverage weights and the corrected root/surface storage ratio
    print("\n[7] NLDAS coverage and root-zone sanity")

    ok = True

    valid = data["nldas_n_valid_pixels"]
    total = data["nldas_n_total_pixels"]

    if (total.dropna() <= 0).any():
        count = int((total <= 0).sum())
        print(
            f"  FAIL: {count:,} rows have zero/negative total NLDAS coverage"
        )
        ok = False
    else:
        print("  PASS: total NLDAS coverage is positive")

    coverage_bad = (
        valid.notna()
        & total.notna()
        & ((valid < -1e-9) | (valid > total + 1e-6))
    )

    if coverage_bad.any():
        print(
            f"  FAIL: {int(coverage_bad.sum()):,} rows have invalid "
            "valid/total coverage relationships"
        )
        ok = False
    else:
        print("  PASS: valid coverage is between 0 and total coverage")

    surface_mean = float(
        np.nanmean(data["soilm_surface_mean"])
    )
    root_mean = float(
        np.nanmean(data["soilm_root_0_100_mean"])
    )

    if not np.isfinite(surface_mean) or surface_mean <= 0:
        print("  FAIL: surface soil mean is not usable for ratio check")
        return False

    root_ratio = root_mean / surface_mean

    print(
        f"  INFO: 0-100/surface mean ratio = {root_ratio:.2f}x"
    )
   

    if root_ratio < 5.0:
        print(
            "  FAIL: 0-100 cm storage is too close to the surface layer; "
            "the wrong NLDAS variable may be bound"
        )
        ok = False
    else:
        print("  PASS: corrected 0-100 cm storage behaves like a deep layer")


    return ok


def validate(
    csv_path,
    start_year=None,
    end_year=None,
):
    #run all checks and return True only if every required check passes
    data = load_data(csv_path)

    print("=" * 68)
    print("DroughtWatch-MA simplified zonal-statistics validation")
    print(f"File: {csv_path}")
    print(
        f"Rows: {len(data):,} | "
        f"Dates: {data['date'].min().date()} to {data['date'].max().date()}"
    )
    print("=" * 68)

    checks = [
        check_schema(data),
        check_identifiers(data),
    ]

    if not all(checks):
        print("\nVALIDATION FAILED: fix schema/identifier problems first.")
        return False

    checks.extend(
        [
            check_dates_and_rows(
                data,
                start_year=start_year,
                end_year=end_year,
            ),
            check_prism_completeness(data),
            check_numeric_values(data),
            check_missing_values(data),
            check_nldas_coverage(data),
        ]
    )

    passed = all(checks)

    print("\n" + "=" * 68)

    if passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")

    print("=" * 68)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate the simplified DroughtWatch zonal-statistics master"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year that must be complete. Default: first year in the CSV.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year that must be complete. Default: last year in the CSV.",
    )

    args = parser.parse_args()

    if (
        args.start_year is not None
        and args.end_year is not None
        and args.end_year < args.start_year
    ):
        raise ValueError("end-year must be >= start-year")

    passed = validate(
        args.csv,
        args.start_year,
        args.end_year,
    )

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
