#Downloads weekly county DSCI for: Maryland, Virginia, District of Columbia
#2000-2025
#DSCI ranges from 0 to 500 and is used as the calibration target for the composite weights.


import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "usdm"
    / "usdm_county_dsci_2000_2025.csv"
)

API_URL = (
    "https://usdmdataservices.unl.edu/"
    "api/CountyStatistics/GetDSCI"
)

STATES = ["MD", "VA", "DC"]

REQUIRED_COLUMNS = {
    "state",
    "county",
    "fips",
    "mapDate",
    "dsci",
}


COLUMN_NAMES = {
    "state": "state",
    "county": "county",
    "fips": "fips",
    "mapdate": "mapDate",
    "dsci": "dsci",
}


def parse_response(text):
    #convert either JSON or CSV text from the USDM service into a DataFrame.
    stripped = text.lstrip()

    if stripped.startswith("[") or stripped.startswith("{"):
        data = pd.read_json(StringIO(text))
    else:
        data = pd.read_csv(StringIO(text))

    return data


def clean_frame(data, year, state):
    #check the API response and standardize the fields we need.
    #rename State to state, MapDate to mapDate, etc
    data = data.rename(
        columns={
            column: COLUMN_NAMES[column.lower()]
            for column in data.columns
            if column.lower() in COLUMN_NAMES
        }
    )

    missing = REQUIRED_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            f"{year} {state}: API response is missing columns "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            f"{year} {state}: API returned zero rows"
        )

    cleaned = data[
        ["state", "county", "fips", "mapDate", "dsci"]
    ].copy()

    #FIPS is an identifier
    cleaned["fips"] = (
        cleaned["fips"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

   #convert to text first to be read by pandas
    date_text = (
        cleaned["mapDate"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    if date_text.str.fullmatch(r"\d{8}").all():
        cleaned["mapDate"] = pd.to_datetime(
            date_text,
            format="%Y%m%d",
        )
    else:
        cleaned["mapDate"] = pd.to_datetime(date_text)

    #checks that maps what is expected from USDM releases
    if not (cleaned["mapDate"].dt.dayofweek == 1).all():
        raise ValueError(
            f"{year} {state}: found MapDate values that are not Tuesdays"
        )

    cleaned["dsci"] = pd.to_numeric(
        cleaned["dsci"]
    )

    return cleaned


#download one state's county DSCI for one calendar year
def download_one_year_state(
    session,
    year,
    state,
    retries=4,
):
    
    params = {
        "aoi": state,
        "startdate": f"1/1/{year}",
        "enddate": f"12/31/{year}",
        "statisticsType": 1,
    }

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            data = parse_response(
                response.text
            )

            return clean_frame(
                data,
                year,
                state,
            )

        except Exception as error:
            if attempt == retries:
                raise RuntimeError(
                    f"{year} {state} failed after "
                    f"{retries} attempts: {error}"
                ) from error

            time.sleep(2 ** attempt)


#download and combine all requested county DSCI data.
def download_dsci(
    start_year,
    end_year,
    output_file,
):
  
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()
    frames = []

    for year in range(
        start_year,
        end_year + 1,
    ):
        for state in STATES:
            data = download_one_year_state(
                session,
                year,
                state,
            )

            frames.append(data)

            time.sleep(0.3)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    #year-by-year requests return the same weekly record at the 
    #boundary between two years (a request starting 1/1/2024 also
    #returns the 2023-12-26 map)
    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["fips", "mapDate"]
    )

    duplicates_removed = (
        before - len(combined)
    )

    #keep the output in order
    combined = combined.sort_values(
        ["fips", "mapDate"]
    ).reset_index(drop=True)

    #write dates YYYY-MM-DD
    combined["mapDate"] = (
        combined["mapDate"]
        .dt.strftime("%Y-%m-%d")
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print(
        f"DSCI rows: {len(combined):,} "
        f"({combined['fips'].nunique()} counties, "
        f"{combined['mapDate'].min()} to {combined['mapDate'].max()}; "
        f"{duplicates_removed:,} year-boundary duplicates removed)"
    )
    print("\nDone. Output saved to:")
    print(f"  {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download weekly county DSCI "
            "for MD, VA, and DC"
        )
    )

    parser.add_argument(
        "start_year",
        type=int,
        nargs="?",
        default=2000,
    )

    parser.add_argument(
        "end_year",
        type=int,
        nargs="?",
        default=2025,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError(
            "end_year must be greater than or equal to start_year"
        )

    download_dsci(
        args.start_year,
        args.end_year,
        args.output,
    )


if __name__ == "__main__":
    main()
