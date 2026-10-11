#convert raw weather values into z-score anomalies.
#calculate SPI-30, SPI-60, and SPI-90 from precipitation.
#combine eight drought signals into one composite drought score.


from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZONAL_FILE = PROJECT_ROOT / "output" / "zonal_stats_master_ee.csv"
BASELINE_FILE = PROJECT_ROOT / "output" / "baselines.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "features_master.csv"

BASELINE_START = pd.Timestamp("1991-01-01")
BASELINE_END = pd.Timestamp("2020-12-31")

SPI_WINDOWS = [30, 60, 90]

Z_SCORE_VARIABLES = [
    "tmax_mean",
    "tmin_mean",
    "soilm_surface_mean",
    "soilm_root_0_100_mean",
    "et_mean",
]

def load_data():
    #load the daily county data and the county-month baseline table
    
    daily = pd.read_csv(
        ZONAL_FILE,
        parse_dates=["date"],
        dtype={"county_fips": str},
    )
    daily["county_fips"] = daily["county_fips"].str.zfill(5)

    needed = Z_SCORE_VARIABLES + ["precip_mean"]
    missing = [c for c in needed if c not in daily.columns]
    if missing:
        raise ValueError(f"Daily data is missing columns: {missing}")

    baselines = pd.read_csv(
        BASELINE_FILE,
        dtype={"county_fips": str},
    )
    baselines["county_fips"] = baselines["county_fips"].str.zfill(5)

    if baselines.duplicated(["county_fips", "month"]).any():
        raise ValueError("baselines.csv has duplicate county/month rows.")

    daily = daily.sort_values(["county_fips", "date"]).reset_index(drop=True)
    daily["month"] = daily["date"].dt.month

    return daily, baselines


def check_daily_dates(data):
    #make sure every county has one row per consecutive day
    
    for county_fips, group in data.groupby("county_fips"):
        gaps = group["date"].diff().dropna()

        if not (gaps == pd.Timedelta(days=1)).all():
            raise ValueError(
                f"County {county_fips} has missing dates. "
                "Fix the daily input before using rolling windows."
            )


def add_z_scores(data, baselines):
    #compare each daily value with what is normal for that county and month
    #z = (observed - normal_mean) / normal_standard_deviation
   
    data = data.merge(
        baselines,
        on=["county_fips", "month"],
        how="left",
        validate="many_to_one",
    )

    for variable in Z_SCORE_VARIABLES:
        mean_col = f"{variable}_baseline_mean"
        std_col = f"{variable}_baseline_std"
        z_col = f"{variable}_z"

        data[z_col] = (
            (data[variable] - data[mean_col])
            / data[std_col]
        )

        #divide-by-zero can create +inf or -inf
        data[z_col] = data[z_col].replace([np.inf, -np.inf], np.nan)

    baseline_columns = [
        c for c in data.columns
        if "_baseline_mean" in c or "_baseline_std" in c
    ]

    return data.drop(columns=baseline_columns)


def fit_spi_distribution(values):
    #fit the precipitation distribution used by SPI
    #SPI uses a gamma distribution for positive accumulated precipitation
    #and separately keeps track of how often accumulation is exactly zero

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 10:
        return None

    probability_zero = np.mean(values == 0)
    positive = values[values > 0]

    if len(positive) < 10:
        return None

    shape, location, scale = stats.gamma.fit(positive, floc=0)

    return shape, location, scale, probability_zero


def convert_to_spi(values, fitted):
    #convert accumulated precipitation values into SPI values
    
    shape, location, scale, probability_zero = fitted

    values = np.asarray(values, dtype=float)
    spi = np.full(len(values), np.nan)

    good = np.isfinite(values)

    if not good.any():
        return spi

    x = values[good]

    gamma_probability = stats.gamma.cdf(
        x,
        shape,
        loc=location,
        scale=scale,
    )

    total_probability = (
        probability_zero
        + (1 - probability_zero) * gamma_probability
    )

    #exact zero precipitation uses the observed probability of zero
    total_probability[x == 0] = probability_zero

    #avoid infinite SPI at probabilities exactly 0 or 1
    total_probability = np.clip(
        total_probability,
        1e-6,
        1 - 1e-6,
    )

    spi[good] = stats.norm.ppf(total_probability)
    return spi


def add_spi(data):
    #calculate SPI-30, SPI-60, and SPI-90
    #rolling precipitation total
    #for each county and calendar month, fit the gamma distribution using only 1991-2020
    #convert every available rolling total to a standard-normal SPI
    
    baseline = (
        (data["date"] >= BASELINE_START)
        & (data["date"] <= BASELINE_END)
    )

    for window in SPI_WINDOWS:
        rolling_col = f"_precip_{window}d"
        spi_col = f"spi_{window}"

        data[rolling_col] = (
            data.groupby("county_fips")["precip_mean"]
            .transform(
                lambda s: s.rolling(
                    window,
                    min_periods=window,
                ).sum()
            )
        )

        data[spi_col] = np.nan

        for county_fips, county_rows in data.groupby("county_fips"):
            county_index = county_rows.index

            for month in range(1, 13):
                month_index = county_index[
                    data.loc[county_index, "month"] == month
                ]

                fit_index = month_index[
                    baseline.loc[month_index]
                ]

                fit_values = (
                    data.loc[fit_index, rolling_col]
                    .dropna()
                    .to_numpy()
                )

                fitted = fit_spi_distribution(fit_values)

                if fitted is None:
                    continue

                apply_values = data.loc[month_index, rolling_col].to_numpy()

                data.loc[month_index, spi_col] = convert_to_spi(
                    apply_values,
                    fitted,
                )

        data = data.drop(columns=[rolling_col])

    return data


def choose_output_columns(data):
    columns = [
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
        "et_mean",
    ]

    columns += [f"{v}_z" for v in Z_SCORE_VARIABLES]
    columns += ["spi_30", "spi_60", "spi_90"]
    columns += [
        "nldas_n_valid_pixels",
        "nldas_n_total_pixels",
    ]

    missing = [c for c in columns if c not in data.columns]

    if missing:
        raise ValueError(f"Cannot save features; missing columns: {missing}")

    return data[columns].copy()


def validate_output(data):
    if data.duplicated(["county_fips", "date"]).any():
        raise ValueError("Duplicate county/date rows found.")

    for spi_col in ["spi_30", "spi_60", "spi_90"]:
        finite = data[spi_col].dropna()

        if not finite.empty and finite.abs().max() > 4.9:
            raise ValueError(f"{spi_col} contains an implausibly large value.")

    if np.isinf(data.select_dtypes(include=[np.number]).to_numpy()).any():
        raise ValueError("Infinite numeric values found in feature output.")


def main():
    data, baselines = load_data()

    check_daily_dates(data)

    data = add_z_scores(data, baselines)

    data = data.sort_values(["county_fips", "date"]).reset_index(drop=True)

    data = add_spi(data)

    output = choose_output_columns(data)
    validate_output(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Feature rows: {len(output):,} "
        f"({output['county_fips'].nunique()} counties, "
        f"{output['date'].min().date()} to {output['date'].max().date()})"
    )
    print("\nDone. Output saved to:")
    print(f"  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
