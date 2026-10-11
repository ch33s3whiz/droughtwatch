import argparse
import calendar
import json
import netrc
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#temporary hourly files
RAW_DIR = PROJECT_ROOT / "data" / "nldas_raw_v2"
#finished daily files
DAILY_DIR = PROJECT_ROOT / "data" / "nldas_daily_v2" 

EARTHDATA_HOST = "urs.earthdata.nasa.gov"
COLLECTION = "NLDAS_NOAH0125_H.2.0"
COLLECTION_ID = "C2069246977-GES_DISC"
BASE_URL = f"https://opendap.earthdata.nasa.gov/collections/{COLLECTION_ID}/granules"


VARIABLES = [
    "SoilM_0_100cm", "RootMoist",
    "SoilM_0_10cm", "SoilM_10_40cm", "SoilM_40_100cm", "SoilM_100_200cm",
    "SMAvail_0_100cm",
    "Evap", "PotEvap", "TVeg", "ESoil", "ECanop",
    "Qs", "Qsb", "Rainf", "LAI", "Streamflow",
]

#some variables describe a condition during an hour, so they are averaged
#others describe an amount accumulated during an hour, so they are summed
DAILY_METHOD = {
    "SoilM_0_100cm": "mean", "RootMoist": "mean",
    "SoilM_0_10cm": "mean", "SoilM_10_40cm": "mean",
    "SoilM_40_100cm": "mean", "SoilM_100_200cm": "mean",
    "SMAvail_0_100cm": "mean",
    "Evap": "sum", "PotEvap": "mean", "TVeg": "mean",
    "ESoil": "mean", "ECanop": "mean",
    "Qs": "sum", "Qsb": "sum", "Rainf": "sum",
    "LAI": "mean", "Streamflow": "mean",
}

#rough box around Maryland, DC, and Virginia
BBOX = {"lat_min": 35.9, "lat_max": 40.9, "lon_min": -84.0, "lon_max": -74.6}
PAD_CELLS = 2



#log in to NASA Earthdata
def earthdata_login():
    #read username/password from ~/.netrc.
    try:
        auth = netrc.netrc().authenticators(EARTHDATA_HOST)
    except FileNotFoundError as error:
        raise RuntimeError("Missing ~/.netrc with NASA Earthdata login") from error

    if not auth:
        raise RuntimeError(f"No login for {EARTHDATA_HOST} in ~/.netrc")

    username, _, password = auth
    return username, password


_thread_data = threading.local()


def session(login):
    #return this thread's authenticated requests session
    s = getattr(_thread_data, "session", None)

    if s is None:
        s = requests.Session()
        s.auth = login
        s.headers["User-Agent"] = "DroughtWatch simple NLDAS downloader"
        _thread_data.session = s

    return s


#build NASA OPeNDAP URLs
def granule_name(day, hour):
    return f"NLDAS_NOAH0125_H.A{day.strftime('%Y%m%d')}.{hour:02d}00.020.nc"


def granule_url(day, hour):
    name = urllib.parse.quote(f"{COLLECTION}:{granule_name(day, hour)}", safe="")
    return f"{BASE_URL}/{name}"


def coordinate_url(day):
    expression = urllib.parse.quote("/lat;/lon", safe="")
    return granule_url(day, 12) + f".dap.nc4?dap4.ce={expression}"


def subset_url(day, hour, grid):
    """Ask NASA for only our variables and only our Mid-Atlantic grid cells."""
    lat0, lat1 = grid["lat0"], grid["lat1"]
    lon0, lon1 = grid["lon0"], grid["lon1"]

    parts = [
        f"/{var}[0][{lat0}:{lat1}][{lon0}:{lon1}]"
        for var in VARIABLES
    ]
    parts += [f"/lat[{lat0}:{lat1}]", f"/lon[{lon0}:{lon1}]", "/time"]

    expression = urllib.parse.quote(";".join(parts), safe="")
    return granule_url(day, hour) + f".dap.nc4?dap4.ce={expression}"



#find which NLDAS grid cells cover MD/DC/VA
def get_grid_window(cache_file, day, login):
    """Read NASA's coordinates and find the indexes around our study area."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = cache_file.with_suffix(".nc4")

    r = session(login).get(coordinate_url(day), timeout=120)
    r.raise_for_status()
    temp_file.write_bytes(r.content)

    with xr.open_dataset(temp_file, engine="netcdf4") as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values

    temp_file.unlink(missing_ok=True)

    lat_hits = np.where((lat >= BBOX["lat_min"]) & (lat <= BBOX["lat_max"]))[0]
    lon_hits = np.where((lon >= BBOX["lon_min"]) & (lon <= BBOX["lon_max"]))[0]

    if len(lat_hits) == 0 or len(lon_hits) == 0:
        raise RuntimeError("Study area did not overlap the NLDAS grid")

    grid = {
        "lat0": max(0, int(lat_hits[0]) - PAD_CELLS),
        "lat1": min(len(lat) - 1, int(lat_hits[-1]) + PAD_CELLS),
        "lon0": max(0, int(lon_hits[0]) - PAD_CELLS),
        "lon1": min(len(lon) - 1, int(lon_hits[-1]) + PAD_CELLS),
    }

    cache_file.write_text(json.dumps(grid, indent=2))
    return grid


#download one hourly NLDAS subset
def hour_path(raw_dir, day, hour):
    return raw_dir / f"{granule_name(day, hour)}4" 


def hourly_file_ok(path):
    """A good file must open and contain every variable we requested."""
    if not path.exists() or path.stat().st_size < 500:
        return False

    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            return all(var in ds.data_vars for var in VARIABLES)
    except Exception:
        return False


def download_hour(raw_dir, day, hour, grid, login, retries=6):
    path = hour_path(raw_dir, day, hour)

    if hourly_file_ok(path):
        return path

    path.unlink(missing_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            r = session(login).get(subset_url(day, hour, grid), stream=True, timeout=180)
            r.raise_for_status()

            with temp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if not hourly_file_ok(temp_path):
                raise ValueError("Downloaded NetCDF is incomplete")

            #a file becomes 'finished' only after it passes the check.
            temp_path.replace(path)
            return path

        except Exception:
            temp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(60, 2 ** attempt))

    return None


def download_24_hours(raw_dir, day, grid, login, workers):
    #download missing hours, and then return all valid hourly files
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        hour for hour in range(24)
        if not hourly_file_ok(hour_path(raw_dir, day, hour))
    ]

    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_hour, raw_dir, day, hour, grid, login)
                for hour in missing
            ]
            for future in futures:
                future.result()

    return [
        hour_path(raw_dir, day, hour)
        for hour in range(24)
        if hourly_file_ok(hour_path(raw_dir, day, hour))
    ]



#combine the hourly files into one daily NetCDF
def daily_path(daily_dir, day):
    return daily_dir / f"NLDAS_daily_{day.strftime('%Y%m%d')}.nc"


def daily_file_ok(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False

    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            return all(var.upper() in ds.data_vars for var in VARIABLES)
    except Exception:
        return False


def make_daily_file(daily_dir, day, hourly_files):
    #for each grid cell, average or sum its valid hourly values
    if len(hourly_files) != 24:
        return False

    output = daily_path(daily_dir, day)
    temp_output = output.with_suffix(".nc.tmp")
    temp_output.unlink(missing_ok=True)

    totals = {}
    counts = {}
    attrs = {}
    lat = lon = dims = None

    try:
        for file_number, path in enumerate(sorted(hourly_files)):
            with xr.open_dataset(path, engine="netcdf4") as ds:
                for var in VARIABLES:
                    data = ds[var]
                    if "time" in data.dims:
                        data = data.isel(time=0, drop=True)

                    values = data.values.astype(float)

                    if file_number == 0:
                        totals[var] = np.zeros(values.shape)
                        counts[var] = np.zeros(values.shape)
                        attrs[var] = dict(data.attrs)
                        if lat is None:
                            lat = ds["lat"].values
                            lon = ds["lon"].values
                            dims = data.dims

                    valid = np.isfinite(values)
                    totals[var][valid] += values[valid]
                    counts[var][valid] += 1

        daily_variables = {}

        for var in VARIABLES:
            if DAILY_METHOD[var] == "mean":
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.where(counts[var] > 0, totals[var] / counts[var], np.nan)
            else:
                values = np.where(counts[var] > 0, totals[var], np.nan)

            var_attrs = attrs[var]
            var_attrs["cell_methods"] = f"time: {DAILY_METHOD[var]}"
            var_attrs.pop("vmin", None)
            var_attrs.pop("vmax", None)

            daily_variables[var.upper()] = (
                dims,
                values.astype("float32"),
                var_attrs,
            )

        daily = xr.Dataset(
            daily_variables,
            coords={"lat": lat, "lon": lon},
            attrs={
                "date": day.strftime("%Y-%m-%d"),
                "source": COLLECTION,
                "hours_used": len(hourly_files),
            },
        )

        daily_dir.mkdir(parents=True, exist_ok=True)
        daily.to_netcdf(temp_output, engine="netcdf4")
        daily.close()

        if not daily_file_ok(temp_output):
            raise ValueError("Daily NetCDF failed validation")

        temp_output.replace(output)
        return True

    except Exception as error:
        temp_output.unlink(missing_ok=True)
        print(f"FAIL  {day.date()}: daily file could not be written ({error})")
        return False



#main download loop
def download_nldas(start_year, end_year, raw_dir, daily_dir, workers):
    login = earthdata_login()
    grid = get_grid_window(
        raw_dir / "grid_indices.json",
        pd.Timestamp(f"{start_year}-06-15"),
        login,
    )

    failures = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_day = calendar.monthrange(year, month)[1]
            days = pd.date_range(
                f"{year}-{month:02d}-01",
                f"{year}-{month:02d}-{last_day}",
            )

            for day in days:
                output = daily_path(daily_dir, day)

                #do nothing if the final daily file is already good
                if daily_file_ok(output):
                    continue

                hourly_files = download_24_hours(raw_dir, day, grid, login, workers)

                if make_daily_file(daily_dir, day, hourly_files):
                    #delete temporary hourly subsets only after daily validation
                    for path in hourly_files:
                        path.unlink(missing_ok=True)
                else:
                    #hourly files are stored, so rerunning the script only downloads the hours that are still missing
                    if len(hourly_files) != 24:
                        print(
                            f"FAIL  {day.date()}: only {len(hourly_files)}/24 "
                            "hours downloaded (rerun to retry)"
                        )
                    failures += 1

    if failures:
        print(f"\nFinished with {failures} failed day(s). Rerun to retry them.")
        raise SystemExit(1)

    print("Done. Output saved to:")
    print(f"  {daily_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and aggregate NLDAS data")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    download_nldas(
        args.start_year,
        args.end_year,
        RAW_DIR,
        DAILY_DIR,
        args.workers,
    )


if __name__ == "__main__":
    main()
