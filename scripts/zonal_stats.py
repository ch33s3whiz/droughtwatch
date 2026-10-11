#Read PRISM rasters and calculate county values
#Read the daily NLDAS file and calculate county values
#Merge both 
#Save one row per county per day

import argparse
import calendar
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract

from nldas_ee_core import (
    check_grid,
    extract_one_day,
    open_nldas_day,
    sanitize_da,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COUNTY_FILE = PROJECT_ROOT / "data" / "shape_files" / "mid_atlantic_counties.shp"

DEFAULT_PRISM_DIR = PROJECT_ROOT / "data" / "prism_raw"
DEFAULT_NLDAS_DIR = PROJECT_ROOT / "data" / "nldas_daily_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "zonal_stats_master_ee.csv"

EXPECTED_COUNTIES = 158

PRISM_VARIABLES = {
    "ppt": "precip",
    "tmax": "tmax",
    "tmin": "tmin",
}

OUTPUT_COLUMNS = [
    "county_fips", "county_name", "state", "date",
    "precip_mean", "precip_std",
    "tmax_mean", "tmax_std",
    "tmin_mean",
    "soilm_surface_mean", "soilm_surface_std",
    "soilm_root_0_100_mean", "soilm_root_0_100_std",
    "soilm_10_40_mean", "soilm_10_40_std",
    "et_mean",
    "nldas_n_valid_pixels", "nldas_n_total_pixels",
    "prism_variables_found",
]


def load_counties():
    #load the 158 MD/DC/VA county polygons
    counties = gpd.read_file(COUNTY_FILE)

    if counties.crs is None:
        raise ValueError("County shapefile has no coordinate system.")

    if "GEOID" not in counties.columns or "NAME" not in counties.columns:
        raise ValueError("County shapefile must contain GEOID and NAME.")

    counties["county_fips"] = counties["GEOID"].astype(str).str.zfill(5)

    if counties["county_fips"].nunique() != EXPECTED_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTIES} counties, "
            f"found {counties['county_fips'].nunique()}."
        )

    state_from_fips = {"11": "DC", "24": "MD", "51": "VA"}

    if "STUSPS" in counties.columns:
        counties["state"] = counties["STUSPS"]
    else:
        counties["state"] = counties["county_fips"].str[:2].map(state_from_fips)

    return counties


def prism_for_day(day, counties, prism_dir):
    #turn one day's PRISM rasters into county-level values
    date_text = day.strftime("%Y%m%d")
    output_date = day.strftime("%Y-%m-%d")

    result = counties[["county_fips", "NAME", "state"]].copy()
    result = result.rename(columns={"NAME": "county_name"})
    result["date"] = output_date

    found = []

    for prism_name, output_name in PRISM_VARIABLES.items():
        folder = prism_dir / prism_name / f"{prism_name}_{date_text}"
        files = list(folder.glob("*.bil"))

        if not files:
            print(f"Missing PRISM {prism_name} for {output_date}")
            continue

        raster = files[0]
        found.append(prism_name)

        with rasterio.open(raster) as src:
            raster_crs = src.crs

        prism_counties = counties.to_crs(raster_crs)

        if prism_name in {"ppt", "tmax"}:
            operations = ["mean", "stdev"]
        else:
            operations = ["mean"]

        stats = exact_extract(
            str(raster), 
            prism_counties,
            operations,
            include_cols=["county_fips"],
            output="pandas",
        )

        rename = {"mean": f"{output_name}_mean"}

        if "stdev" in stats.columns:
            rename["stdev"] = f"{output_name}_std"

        stats = stats.rename(columns=rename)

        result = result.merge(
            stats,
            on="county_fips",
            how="left",
            validate="one_to_one",
        )

    result["prism_variables_found"] = ",".join(found)
    return result


def nldas_for_day(day, counties, nldas_dir, do_first_day_checks=False):
    #turn one daily NLDAS NetCDF into county-level values
    date_text = day.strftime("%Y%m%d")
    output_date = day.strftime("%Y-%m-%d")
    nc_file = nldas_dir / f"NLDAS_daily_{date_text}.nc"

    if not nc_file.exists():
        print(f"Missing NLDAS file for {output_date}")
        return None

    nldas_counties = counties[["county_fips", "geometry"]].to_crs("EPSG:4326")

    ds, var_map = open_nldas_day(nc_file)

    try:
        if do_first_day_checks:
            sample = sanitize_da(ds, var_map["et"])
            findings = check_grid(sample)

            if findings["problems"]:
                raise ValueError(
                    "NLDAS grid check failed: "
                    + "; ".join(findings["problems"])
                )

        result = extract_one_day(
            ds,
            nldas_counties,
            var_map,
            output_date,
            check_ratio=do_first_day_checks,
        )

    finally:
        ds.close()

    return result


def process_day(day, counties, prism_dir, nldas_dir, first_nldas_check):
    #process one day and merge PRISM with NLDAS
    prism = prism_for_day(day, counties, prism_dir)

    #if every PRISM raster is missing skip this day
    if prism["prism_variables_found"].iloc[0] == "":
        return None, first_nldas_check

    nldas = nldas_for_day(
        day,
        counties,
        nldas_dir,
        do_first_day_checks=first_nldas_check,
    )

    if nldas is None:
        return prism, first_nldas_check

    result = prism.merge(
        nldas,
        on=["county_fips", "date"],
        how="left",
        validate="one_to_one",
    )

    #the grid/root-zone checks are only run on first usable NLDAS day
    return result, False


def run(start_year, end_year, prism_dir, nldas_dir, output_file, append=False):
    #process a chosen range of years
    if not COUNTY_FILE.exists():
        raise FileNotFoundError(f"Missing county shapefile: {COUNTY_FILE}")

    if not prism_dir.exists():
        raise FileNotFoundError(f"Missing PRISM directory: {prism_dir}")

    if not nldas_dir.exists():
        raise FileNotFoundError(f"Missing NLDAS directory: {nldas_dir}")

    if output_file.exists() and not append:
        raise FileExistsError(
            f"{output_file} already exists. "
            "Delete it, rename it, or use --append."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    counties = load_counties()
    first_nldas_check = True
    write_header = not output_file.exists()
    rows_written = 0

    for year in range(start_year, end_year + 1):
        year_parts = []

        for day in pd.date_range(f"{year}-01-01", f"{year}-12-31"):
            day_result, first_nldas_check = process_day(
                day,
                counties,
                prism_dir,
                nldas_dir,
                first_nldas_check,
            )

            if day_result is not None:
                year_parts.append(day_result)

        if not year_parts:
            print(f"No rows produced for {year}.")
            continue

        year_data = pd.concat(year_parts, ignore_index=True)

        #guarantee the same CSV schema every year
        for column in OUTPUT_COLUMNS:
            if column not in year_data.columns:
                year_data[column] = pd.NA

        year_data = year_data[OUTPUT_COLUMNS]

        if output_file.exists() and not write_header:
            old_columns = pd.read_csv(output_file, nrows=0).columns.tolist()
            if old_columns != OUTPUT_COLUMNS:
                raise ValueError(
                    "Existing output CSV has a different column layout. "
                    "Refusing to append."
                )

        days = 366 if calendar.isleap(year) else 365
        expected_rows = EXPECTED_COUNTIES * days

        if len(year_data) != expected_rows:
            print(
                f"WARNING: {year} produced {len(year_data):,} rows; "
                f"expected {expected_rows:,}"
            )

        rows_written += len(year_data)

        year_data.to_csv(
            output_file,
            mode="a",
            header=write_header,
            index=False,
        )

        write_header = False

    print(
        f"County-day rows written: {rows_written:,} "
        f"({start_year}-{end_year})"
    )
    print("\nDone. Output saved to:")
    print(f"  {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple DroughtWatch zonal statistics"
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("--prism-dir", type=Path, default=DEFAULT_PRISM_DIR)
    parser.add_argument("--nldas-dir", type=Path, default=DEFAULT_NLDAS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    run(
        args.start_year,
        args.end_year,
        args.prism_dir,
        args.nldas_dir,
        args.output,
        append=args.append,
    )


if __name__ == "__main__":
    main()
