#Turn every feature so positive means more drought stress
#Average daily features into USDM-style weeks ending Tuesday
#Join county-week features to county-week USDM DSCI
#Fit nonnegative feature coefficients with a free intercept
#Normalize the eight feature coefficients so they sum to 1


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = PROJECT_ROOT / "output" / "features_master.csv"
USDM_FILE = PROJECT_ROOT / "data" / "usdm" / "usdm_county_dsci_2000_2025.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"

WEIGHTS_FILE = OUTPUT_DIR / "phase4_calibrated_weights.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase4_fit_summary.json"
DIAGNOSTICS_FILE = OUTPUT_DIR / "phase4_calibration_diagnostics.csv"
MODEL_DATASET_FILE = OUTPUT_DIR / "model_dataset_2000_2025.csv"

CALIB_START = pd.Timestamp("2000-01-04")
CALIB_END = pd.Timestamp("2018-12-31")
HOLDOUT_START = pd.Timestamp("2019-01-01")

#the eight drought-aligned columns used by the model.
#_neg columns have their sign flipped so that larger = drier
FEATURE_COLS = [
    "spi_30_neg",
    "spi_60_neg",
    "spi_90_neg",
    "soilm_surface_neg",
    "soilm_root_0_100_neg",
    "et_neg",
    "tmax_mean_z",
    "tmin_mean_z",
]

#the names Phase 3 looks up when it builds composite_score from this weight table
WEIGHT_NAMES = [
    "spi_30",
    "spi_60",
    "spi_90",
    "soilm_surface_mean_z",
    "soilm_root_0_100_mean_z",
    "et_mean_z",
    "tmax_mean_z",
    "tmin_mean_z",
]

LABELS = [
    "SPI-30",
    "SPI-60",
    "SPI-90",
    "Surface soil moisture",
    "Root soil moisture (0-100 cm)",
    "Actual ET",
    "Tmax",
    "Tmin",
]


def load_features():
    #load daily Phase 3 features and create columns
    data = pd.read_csv(
        FEATURES_FILE,
        parse_dates=["date"],
        dtype={"county_fips": str},
    )

    data["county_fips"] = data["county_fips"].str.zfill(5)

    required = [
        "spi_30",
        "spi_60",
        "spi_90",
        "soilm_surface_mean_z",
        "soilm_root_0_100_mean_z",
        "et_mean_z",
        "tmax_mean_z",
        "tmin_mean_z",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"features_master.csv is missing: {missing}")

    #flip signs so that larger number = more drought stress for every feature
    data["spi_30_neg"] = -data["spi_30"]
    data["spi_60_neg"] = -data["spi_60"]
    data["spi_90_neg"] = -data["spi_90"]
    data["soilm_surface_neg"] = -data["soilm_surface_mean_z"]
    data["soilm_root_0_100_neg"] = -data["soilm_root_0_100_mean_z"]
    data["et_neg"] = -data["et_mean_z"]

    return data


def load_usdm():
    #load USDM DSCI and standardize common column-name variations.
    #DSCI ranges from 0 to 500 so dividing by 500 puts the target on a 0-to-1 scale
    raw = pd.read_csv(USDM_FILE, dtype=str)

    date_col = next(
        (c for c in ["MapDate", "release_date", "date", "mapDate"] if c in raw.columns),
        None,
    )
    fips_col = next(
        (c for c in ["county_fips", "FIPS", "fips"] if c in raw.columns),
        None,
    )
    dsci_col = next(
        (c for c in ["DSCI", "dsci"] if c in raw.columns),
        None,
    )

    if date_col is None or fips_col is None or dsci_col is None:
        raise ValueError(
            "USDM CSV needs a date column, a FIPS column, and DSCI."
        )

    date_text = raw[date_col].astype(str)

    if date_text.str.fullmatch(r"\d{8}").all():
        release_date = pd.to_datetime(date_text, format="%Y%m%d")
    else:
        release_date = pd.to_datetime(date_text)

    usdm = pd.DataFrame(
        {
            "county_fips": raw[fips_col].astype(str).str.zfill(5),
            "release_date": release_date,
            "DSCI": pd.to_numeric(raw[dsci_col]),
        }
    )

    usdm["dsci_normalized"] = usdm["DSCI"] / 500.0

    return usdm


def make_weekly_features(data, columns):
    #average daily feature values into weeks ending on Tuesday to line up the county-week 
    #rows with the Tuesday release dates used by the U.S. Drought Monitor
    weekly = (
        data.sort_values(["county_fips", "date"])
        .set_index("date")
        .groupby("county_fips")[columns]
        .resample("W-TUE")
        .mean()
        .reset_index()
        .rename(columns={"date": "release_date"})
    )

    return weekly


def build_calibration_data(features, usdm):
    #build the 2000-2018 county-week table used to learn the weights.
    #include six days before the first Tuesday so the first weekly bin can contain a complete seven-day window
    start_for_daily_data = CALIB_START - pd.Timedelta(days=6)

    daily = features[
        (features["date"] >= start_for_daily_data)
        & (features["date"] <= CALIB_END)
    ].copy()

    weekly = make_weekly_features(daily, FEATURE_COLS)

    calibration_usdm = usdm[
        (usdm["release_date"] >= CALIB_START)
        & (usdm["release_date"] <= CALIB_END)
    ]

    joined = weekly.merge(
        calibration_usdm,
        on=["county_fips", "release_date"],
        how="inner",
        validate="one_to_one",
    )

    return joined.dropna(
        subset=FEATURE_COLS + ["dsci_normalized"]
    ).copy()


def load_calibration_from_model_dataset():
    if not MODEL_DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing model dataset: {MODEL_DATASET_FILE}")

    data = pd.read_csv(
        MODEL_DATASET_FILE,
        parse_dates=["release_date"],
        dtype={"county_fips": str},
    )

    data["county_fips"] = data["county_fips"].str.zfill(5)

    needed = (
        ["county_fips", "release_date"]
        + FEATURE_COLS
        + ["DSCI", "dsci_normalized"]
    )

    missing = [c for c in needed if c not in data.columns]

    if missing:
        raise ValueError(f"Model dataset is missing columns: {missing}")

    calibration = data[
        (data["release_date"] >= CALIB_START)
        & (data["release_date"] <= CALIB_END)
    ][needed]

    return calibration.dropna(
        subset=FEATURE_COLS + ["dsci_normalized"]
    ).copy()


def solve_weights(calibration):
    #DSCI = feature1*b1 + ... + feature8*b8 + intercept
    #every feature coefficient must be >= 0 
    #intercept may be positive or negative
    #after fitting, divide feature coefficients by their sum so the
    #eight final composite weights add to 1.
    
    X = calibration[FEATURE_COLS].to_numpy(dtype=float)
    y = calibration["dsci_normalized"].to_numpy(dtype=float)

    #add a final column of 1s
    #its coefficient becomes the intercept
    design = np.column_stack([X, np.ones(len(X))])

    lower_bounds = np.r_[
        np.zeros(len(FEATURE_COLS)),
        -np.inf,
    ]

    upper_bounds = np.r_[
        np.full(len(FEATURE_COLS), np.inf),
        np.inf,
    ]

    result = lsq_linear(
        design,
        y,
        bounds=(lower_bounds, upper_bounds),
    )

    raw_weights = result.x[:-1]
    intercept = float(result.x[-1])

    total = raw_weights.sum()

    if total <= 0:
        raise RuntimeError("Calibration produced no usable weights.")

    normalized_weights = raw_weights / total

    return raw_weights, normalized_weights, intercept


def add_predictions(
    calibration,
    raw_weights,
    normalized_weights,
    intercept,
):
    #both forms of the fitted model.
    #calibrated_composite_index: the final weighted drought index
    #fitted_dsci_normalized:the same index scaled to the DSCI target for diagnostics

    X = calibration[FEATURE_COLS].to_numpy(dtype=float)

    calibration = calibration.copy()

    calibration["calibrated_composite_index"] = (
        X @ normalized_weights
    )

    calibration["fitted_dsci_normalized"] = (
        X @ raw_weights + intercept
    )

    calibration["fitted_dsci"] = (
        calibration["fitted_dsci_normalized"] * 500.0
    )

    return calibration


def fit_metrics(data):
    actual = data["dsci_normalized"].to_numpy()
    predicted = data["fitted_dsci_normalized"].to_numpy()

    error = predicted - actual

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    if len(actual) > 1:
        correlation = np.corrcoef(actual, predicted)[0, 1]
    else:
        correlation = np.nan

    return rmse, mae, correlation


def save_weights(raw_weights, normalized_weights):
    table = pd.DataFrame(
        {
            "feature_label": LABELS,
            "weight_name": WEIGHT_NAMES,
            "feature_column": FEATURE_COLS,
            "raw_weight": raw_weights,
            "normalized_weight": normalized_weights,
            "percent_of_total": normalized_weights * 100,
        }
    )

    table.to_csv(WEIGHTS_FILE, index=False)
    return WEIGHTS_FILE


def save_fit_summary(
    raw_weights,
    intercept,
    n_rows,
    rmse,
    mae,
    correlation,
):
    
    summary = {
        "intercept": float(intercept),
        "raw_weight_sum": float(np.sum(raw_weights)),
        "calibration_rows": int(n_rows),
        "calibration_start": str(CALIB_START.date()),
        "calibration_end": str(CALIB_END.date()),
        "in_sample_rmse": float(rmse),
        "in_sample_mae": float(mae),
        "in_sample_pearson": float(correlation),
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return SUMMARY_FILE


def export_model_dataset(features, usdm):
    #create the common weekly table used by later model phases.
    columns = list(FEATURE_COLS)

    if "nldas_n_valid_pixels" in features.columns:
        columns.append("nldas_n_valid_pixels")

    weekly = make_weekly_features(
        features[
            features["date"] >= CALIB_START - pd.Timedelta(days=6)
        ],
        columns,
    )

    weekly = weekly[
        (weekly["release_date"] >= CALIB_START)
        & (weekly["release_date"] <= features["date"].max())
    ].copy()

    if "state" in features.columns:
        state_map = (
            features.drop_duplicates("county_fips")
            .set_index("county_fips")["state"]
        )

        weekly.insert(
            1,
            "state",
            weekly["county_fips"].map(state_map),
        )

    dataset = weekly.merge(
        usdm[
            [
                "county_fips",
                "release_date",
                "DSCI",
                "dsci_normalized",
            ]
        ],
        on=["county_fips", "release_date"],
        how="left",
        validate="one_to_one",
    )

    if "nldas_n_valid_pixels" in dataset.columns:
        dataset = dataset.rename(
            columns={
                "nldas_n_valid_pixels": "nldas_coverage_weight_sum"
            }
        )

    dataset.to_csv(MODEL_DATASET_FILE, index=False)
    return MODEL_DATASET_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Simple DroughtWatch Phase 4 calibration"
    )

    parser.add_argument(
        "--from-model-dataset",
        action="store_true",
        help=(
            "Fit from output/model_dataset_2000_2025.csv instead of the "
            "daily features + USDM files."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_model_dataset:
        calibration = load_calibration_from_model_dataset()
    else:
        features = load_features()
        usdm = load_usdm()
        calibration = build_calibration_data(features, usdm)

    if calibration.empty:
        raise RuntimeError("Calibration table is empty.")

    raw, normalized, intercept = solve_weights(calibration)

    diagnostics = add_predictions(
        calibration,
        raw,
        normalized,
        intercept,
    )

    rmse, mae, correlation = fit_metrics(diagnostics)

    saved = [
        save_weights(raw, normalized),
        save_fit_summary(
            raw,
            intercept,
            len(calibration),
            rmse,
            mae,
            correlation,
        ),
    ]

    diagnostics.to_csv(DIAGNOSTICS_FILE, index=False)
    saved.append(DIAGNOSTICS_FILE)

    
    if not args.from_model_dataset:
        saved.append(export_model_dataset(features, usdm))

    print(f"Calibration rows: {len(calibration):,}")

    print("\nNormalized weights:")
    for label, weight in zip(LABELS, normalized):
        print(f"  {label:<30} {weight:.4f}")

    print(f"\nIntercept: {intercept:.4f}")
    print(f"RMSE:      {rmse:.4f} normalized DSCI ({rmse * 500:.1f} DSCI points)")
    print(f"MAE:       {mae:.4f} normalized DSCI ({mae * 500:.1f} DSCI points)")
    print(f"Pearson r: {correlation:.4f}")

    print("\nDone. Output saved to:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
