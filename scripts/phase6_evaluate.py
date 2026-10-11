#holdout begins in 2019 and ends 03-25-2025
#evaluates NNLS composite prediction and Random Forest prediction

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_FILE = OUTPUT_DIR / "model_dataset_2000_2025.csv"
WEIGHTS_FILE = OUTPUT_DIR / "phase4_calibrated_weights.csv"
FIT_SUMMARY_FILE = OUTPUT_DIR / "phase4_fit_summary.json"
CALIBRATION_FILE = OUTPUT_DIR / "phase4_calibration_diagnostics.csv"
MODEL_FILE = MODEL_DIR / "rf_droughtwatch_2000_2018.joblib"

METRICS_FILE = OUTPUT_DIR / "phase6_holdout_metrics.csv"
CASE_FILE = OUTPUT_DIR / "phase6_case_study.csv"
SEASONAL_FILE = OUTPUT_DIR / "phase6_seasonal.csv"
THRESHOLDS_FILE = OUTPUT_DIR / "phase6_thresholds.csv"
WEEKLY_FILE = OUTPUT_DIR / "phase6_weekly_series.csv"

PERMUTATION_FILE = OUTPUT_DIR / "phase5_permutation_importances.csv"
COMPARISON_FILE = OUTPUT_DIR / "importance_comparison.csv"

CALIBRATION_START = pd.Timestamp("2000-01-04")
CALIBRATION_END = pd.Timestamp("2018-12-31")

HOLDOUT_START = pd.Timestamp("2019-01-01")
HOLDOUT_END = pd.Timestamp("2025-03-25")

EVENT_START = pd.Timestamp("2024-07-01")
RECOVERY_START = pd.Timestamp("2024-10-01")

CORE_START = pd.Timestamp("2024-10-01")
CORE_END = pd.Timestamp("2025-01-31")
CORE_DSCI_MIN = 300.0

#peak weeks from the Phase 6 analysis
PEAK_ALL_COUNTIES = pd.Timestamp("2024-12-03")
PEAK_CORE_COUNTIES = pd.Timestamp("2024-11-26")

ONSET_DSCI = 100.0 / 500.0
PEAK_DSCI = 250.0 / 500.0

THRESHOLD_SENSITIVITY = 0.01
MIN_STRATUM_ROWS = 500
EXPECTED_HOLDOUT_WEEKS = 326

PERMUTATION_REPEATS = 30
RANDOM_SEED = 42
GATE_TOLERANCE = 1e-9

SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def feature_columns():
    return [
        "spi_30_neg",
        "spi_60_neg",
        "spi_90_neg",
        "soilm_surface_neg",
        "soilm_root_0_100_neg",
        "et_neg",
        "tmax_mean_z",
        "tmin_mean_z",
    ]


def required_file(path):
    #return path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return path


def load_inputs():
    #load the model dataset, Phase 4 weights, and RF model
    features = feature_columns()

    dataset_path = required_file(DATASET_FILE)

    weights_path = required_file(WEIGHTS_FILE)

    model_path = required_file(MODEL_FILE)

    data = pd.read_csv(
        dataset_path,
        parse_dates=["release_date"],
        dtype={"county_fips": str},
    )

    required = (
        features
        + [
            "DSCI",
            "dsci_normalized",
        ]
    )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Model dataset is missing: {missing}"
        )

    #any row used for evaluation must have all eight predictors
    target_rows = data[
        data["dsci_normalized"].notna()
    ]

    bad_features = (
        target_rows[features]
        .isna()
        .any(axis=1)
    )

    if bad_features.any():
        raise ValueError(
            f"{int(bad_features.sum()):,} rows "
            "have missing model features."
        )

    weights = pd.read_csv(
        weights_path
    )

    required_weight_columns = {
        "feature_column",
        "raw_weight",
        "normalized_weight",
    }

    if not required_weight_columns.issubset(
        weights.columns
    ):
        raise ValueError(
            "Phase 4 weights file is missing required columns."
        )

    weights = (
        weights.set_index(
            "feature_column"
        )
        .reindex(features)
    )

    if weights[
        ["raw_weight", "normalized_weight"]
    ].isna().any().any():
        raise ValueError(
            "Phase 4 weight rows do not match the selected root layer."
        )

    model = joblib.load(
        model_path
    )

    # A forest fit on a plain array accepts ANY eight columns and returns
    # plausible-looking numbers for the wrong ones. Phase 5 fits on a
    # DataFrame, so the model knows its column names: check them here
    model_columns = list(
        getattr(model, "feature_names_in_", [])
    )

    if model_columns != features:
        raise ValueError(
            "The saved Random Forest was not fit on the expected columns.\n"
            f"Model columns:    {model_columns or 'none recorded'}\n"
            f"Expected columns: {features}\n"
            "Re-run phase5_rf.py."
        )

    return data, weights, model


def load_phase4_fit(
    features,
):
    
    #load the intercept Phase 4 saved, plus the Phase 4 diagnostics table.
    summary_path = required_file(FIT_SUMMARY_FILE)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if "intercept" not in summary:
        raise ValueError(
            f"{summary_path} has no intercept. Re-run Phase 4."
        )

    intercept = float(
        summary["intercept"]
    )

    diagnostics_path = required_file(CALIBRATION_FILE)

    diagnostics = pd.read_csv(
        diagnostics_path
    )

    needed = (
        features
        + [
            "fitted_dsci_normalized",
            "calibrated_composite_index",
        ]
    )

    missing = [
        column
        for column in needed
        if column not in diagnostics.columns
    ]

    if missing:
        raise ValueError(
            f"Calibration diagnostics are missing: {missing}"
        )

    return intercept, diagnostics


 #reproduce the Phase 4 stored index and fitted DSCI before evaluating
def gate_phase4_reproduction(
    diagnostics,
    features,
    normalized_weights,
    raw_weights,
    intercept,
):
    
    X = diagnostics[
        features
    ].to_numpy(dtype=float)

    index = (
        X @ normalized_weights
    )

    fitted = (
        X @ raw_weights
        + intercept
    )

    index_difference = float(
        np.nanmax(
            np.abs(
                index
                - diagnostics[
                    "calibrated_composite_index"
                ].to_numpy(dtype=float)
            )
        )
    )

    fitted_difference = float(
        np.nanmax(
            np.abs(
                fitted
                - diagnostics[
                    "fitted_dsci_normalized"
                ].to_numpy(dtype=float)
            )
        )
    )

    worst = max(
        index_difference,
        fitted_difference,
    )

    if worst >= GATE_TOLERANCE:
        raise ValueError(
            "Phase 4 reproduction failed "
            f"(index diff={index_difference:.3e}, "
            f"fit diff={fitted_difference:.3e}). "
        )


def add_predictions(
    data,
    features,
    normalized_weights,
    raw_weights,
    intercept,
    model,
):
    #add NNLS-composite and RF predictions to the weekly dataset
    data = data.copy()

    X = data[
        features
    ].to_numpy(dtype=float)

    data["composite_index"] = (
        X @ normalized_weights
    )

    data["composite_prediction"] = np.clip(
        X @ raw_weights + intercept,
        0,
        None,
    )

    data["rf_prediction"] = model.predict(
        data[features]
    )

    return data



#calculate pooled and drought-conditional holdout metrics
def metric_rows(
    frame,
    prediction_column,
    model_name,
):
    
    actual_all = frame[
        "dsci_normalized"
    ].to_numpy(dtype=float)

    predicted_all = frame[
        prediction_column
    ].to_numpy(dtype=float)

    strata = [
        (
            "pooled",
            np.ones(
                len(frame),
                dtype=bool,
            ),
        ),
        (
            "conditional_gt0",
            actual_all > 0,
        ),
        (
            "onset_ge_0.20",
            actual_all >= 0.20,
        ),
        (
            "severe_ge_0.60",
            actual_all >= 0.60,
        ),
    ]

    rows = []

    for name, mask in strata:
        actual = actual_all[
            mask
        ]

        predicted = predicted_all[
            mask
        ]

        n = len(actual)

        row = {
            "model": model_name,
            "stratum": name,
            "n": n,
            "usable_for_inference": (
                n >= MIN_STRATUM_ROWS
            ),
            "r2_not_interpretable": (
                name
                in {
                    "onset_ge_0.20",
                    "severe_ge_0.60",
                }
            ),
        }

        if n >= 2 and actual.std() > 0:
            error = (
                predicted - actual
            )

            row.update(
                {
                    "mae": float(
                        np.mean(
                            np.abs(error)
                        )
                    ),
                    "bias_signed": float(
                        error.mean()
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),
                    "r2": float(
                        1
                        - np.sum(
                            error ** 2
                        )
                        / np.sum(
                            (
                                actual
                                - actual.mean()
                            ) ** 2
                        )
                    ),
                    "pearson": float(
                        np.corrcoef(
                            actual,
                            predicted,
                        )[0, 1]
                    ),
                }
            )

        else:
            row.update(
                {
                    "mae": np.nan,
                    "bias_signed": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "pearson": np.nan,
                }
            )

        if name == "pooled":
            zero_mask = (
                actual == 0
            )

            row[
                "zero_dsci_mean_prediction"
            ] = (
                float(
                    predicted[
                        zero_mask
                    ].mean()
                )
                if zero_mask.any()
                else np.nan
            )

        rows.append(row)

    return rows


def anchor_rows(
    holdout,
    date,
    label,
    county_fips=None,
):
    #performance at one fixed drought anchor week
    subset = holdout[
        holdout["release_date"] == date
    ]

    if county_fips is not None:
        subset = subset[
            subset["county_fips"].isin(
                county_fips
            )
        ]

    rows = []

    for prediction, model_name in [
        (
            "composite_prediction",
            "composite",
        ),
        (
            "rf_prediction",
            "rf",
        ),
    ]:
        row = {
            "model": (
                f"{model_name}_at_{label}"
            ),
            "stratum": (
                f"anchor:{label}"
            ),
            "n": len(subset),
            "anchor_week": str(
                date.date()
            ),
            "usable_for_inference": (
                len(subset) > 0
            ),
        }

        if len(subset) > 1:
            actual = (
                subset[
                    "dsci_normalized"
                ]
                .to_numpy(dtype=float)
            )

            predicted = (
                subset[
                    prediction
                ]
                .to_numpy(dtype=float)
            )

            row.update(
                {
                    "pearson": float(
                        np.corrcoef(
                            actual,
                            predicted,
                        )[0, 1]
                    ),
                    "bias_signed": float(
                        (
                            predicted
                            - actual
                        ).mean()
                    ),
                    "mae": float(
                        np.abs(
                            predicted
                            - actual
                        ).mean()
                    ),
                    "mean_prediction": float(
                        predicted.mean()
                    ),
                    "mean_usdm": float(
                        actual.mean()
                    ),
                }
            )

        rows.append(row)

    return rows


def matched_onset_thresholds(
    calibration_weekly,
):
    #Give each model an onset threshold with the same calibration-period
    #exceedance frequency as USDM DSCI > 100.
    
    usdm_frequency = float(
        (
            calibration_weekly[
                "dsci_normalized"
            ]
            > ONSET_DSCI
        ).mean()
    )

    thresholds = {
        "dsci_normalized": ONSET_DSCI
    }

    bands = {
        "dsci_normalized": (
            ONSET_DSCI,
            ONSET_DSCI,
        )
    }

    for column in [
        "composite_prediction",
        "rf_prediction",
    ]:
        thresholds[column] = float(
            calibration_weekly[
                column
            ].quantile(
                1 - usdm_frequency
            )
        )

        bands[column] = (
            float(
                calibration_weekly[
                    column
                ].quantile(
                    1
                    - usdm_frequency
                    - THRESHOLD_SENSITIVITY
                )
            ),
            float(
                calibration_weekly[
                    column
                ].quantile(
                    1
                    - usdm_frequency
                    + THRESHOLD_SENSITIVITY
                )
            ),
        )

    peak_frequency = float(
        (
            calibration_weekly[
                "dsci_normalized"
            ]
            > PEAK_DSCI
        ).mean()
    )

    return (
        thresholds,
        bands,
        usdm_frequency,
        peak_frequency,
    )


def first_exceedance(
    weekly,
    column,
    threshold,
    persistence=1,
):
    #first event-window week above threshold
    event = weekly[
        weekly.index >= EVENT_START
    ]

    above = (
        event[column] > threshold
    ).to_numpy()

    if persistence == 1:
        dates = event.index[
            above
        ]

        return (
            dates.min()
            if len(dates)
            else pd.NaT
        )

    run = np.convolve(
        above.astype(int),
        np.ones(
            persistence,
            dtype=int,
        ),
        mode="valid",
    )

    positions = np.flatnonzero(
        run == persistence
    )

    if len(positions) == 0:
        return pd.NaT

    return event.index[
        positions[0]
    ]


def save_permutation_importance(
    model,
    holdout,
    features,
):
    
    #measure holdout importance by shuffling one feature at a time.
    #scoring metric is RMSE.
    X = holdout[
        features
    ]

    y = holdout[
        "dsci_normalized"
    ].to_numpy()

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scoring=(
            "neg_root_mean_squared_error"
        ),
    )

    table = pd.DataFrame(
        {
            "feature": features,
            "permutation_importance_mean": (
                result.importances_mean
            ),
            "permutation_importance_std": (
                result.importances_std
            ),
        }
    )

    positive = table[
        "permutation_importance_mean"
    ].clip(lower=0)

    total = positive.sum()

    if total > 0:
        table[
            "permutation_importance_normalized"
        ] = positive / total
    else:
        table[
            "permutation_importance_normalized"
        ] = 0.0

    table = table.sort_values(
        "permutation_importance_mean",
        ascending=False,
    )

    path = PERMUTATION_FILE

    table.to_csv(
        path,
        index=False,
    )

    return table, path


#compare NNLS normalized weights with RF holdout permutation importance
def compare_importance(
    weights,
    permutation_table,
    features,
):
    comparison = pd.DataFrame(
        {
            "feature": features,
            "nnls_weight": (
                weights[
                    "normalized_weight"
                ].to_numpy()
            ),
            "rf_permutation_importance": (
                permutation_table
                .set_index("feature")
                .reindex(features)[
                    "permutation_importance_normalized"
                ]
                .to_numpy()
            ),
        }
    )

    comparison["nnls_rank"] = (
        comparison[
            "nnls_weight"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    comparison["rf_rank"] = (
        comparison[
            "rf_permutation_importance"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    comparison["rank_difference"] = (
        comparison["nnls_rank"]
        - comparison["rf_rank"]
    ).abs()

    rho, p_value = spearmanr(
        comparison["nnls_weight"],
        comparison[
            "rf_permutation_importance"
        ],
    )

    comparison[
        "spearman_rho_all_features"
    ] = rho

    comparison[
        "spearman_p_value_all_features"
    ] = p_value

    path = COMPARISON_FILE

    comparison.to_csv(
        path,
        index=False,
    )

    top_nnls = set(
        comparison.nsmallest(
            3,
            "nnls_rank",
        )["feature"]
    )

    top_rf = set(
        comparison.nsmallest(
            3,
            "rf_rank",
        )["feature"]
    )

    top3_overlap = len(
        top_nnls & top_rf
    )

    return path, top3_overlap, rho


#evaluate both models separately by meteorological season
def save_seasonal_metrics(
    holdout,
):
    seasonal_data = holdout.copy()

    seasonal_data["season"] = (
        seasonal_data[
            "release_date"
        ]
        .dt.month
        .map(SEASON)
    )

    rows = []

    for season, group in (
        seasonal_data.groupby("season")
    ):
        actual = group[
            "dsci_normalized"
        ].to_numpy(dtype=float)

        for prediction, model_name in [
            (
                "composite_prediction",
                "composite",
            ),
            (
                "rf_prediction",
                "rf",
            ),
        ]:
            predicted = group[
                prediction
            ].to_numpy(dtype=float)

            error = (
                predicted - actual
            )

            rows.append(
                {
                    "season": season,
                    "model": model_name,
                    "n": len(group),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),
                    "bias_signed": float(
                        error.mean()
                    ),
                    "pearson": float(
                        np.corrcoef(
                            actual,
                            predicted,
                        )[0, 1]
                    ),
                }
            )

    path = SEASONAL_FILE

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


def save_case_study(
    data,
    calibration,
    holdout,
    calibration_weekly,
    holdout_weekly,
    thresholds,
    bands,
    model,
    features,
):
    
    #save main 2024 drought-event findings:
    #onset, peak severity, core counties, recovery, RF ceiling
    
    event = data[
        (data["release_date"] >= CORE_START)
        & (data["release_date"] <= CORE_END)
    ]

    core_counties = (
        event.groupby(
            "county_fips"
        )["DSCI"]
        .max()
    )

    core_counties = (
        core_counties[
            core_counties
            >= CORE_DSCI_MIN
        ]
        .index
        .tolist()
    )

    case = {
        "peak_all_counties": str(
            PEAK_ALL_COUNTIES.date()
        ),
        "peak_core_counties": str(
            PEAK_CORE_COUNTIES.date()
        ),
        "n_core_counties": len(
            core_counties
        ),
    }

    onset = {
        column: first_exceedance(
            holdout_weekly,
            column,
            thresholds[column],
        )
        for column in [
            "dsci_normalized",
            "composite_prediction",
            "rf_prediction",
        ]
    }

    onset_two_weeks = {
        column: first_exceedance(
            holdout_weekly,
            column,
            thresholds[column],
            persistence=2,
        )
        for column in [
            "dsci_normalized",
            "composite_prediction",
            "rf_prediction",
        ]
    }

    labels = {
        "dsci_normalized": "usdm",
        "composite_prediction": "composite",
        "rf_prediction": "rf",
    }

    for column, label in labels.items():
        case[
            f"{label}_onset"
        ] = (
            str(
                onset[column].date()
            )
            if pd.notna(
                onset[column]
            )
            else None
        )

        case[
            f"{label}_onset_persist2"
        ] = (
            str(
                onset_two_weeks[
                    column
                ].date()
            )
            if pd.notna(
                onset_two_weeks[
                    column
                ]
            )
            else None
        )

    for column, label in [
        (
            "composite_prediction",
            "composite",
        ),
        (
            "rf_prediction",
            "rf",
        ),
    ]:
        if (
            pd.notna(
                onset[
                    "dsci_normalized"
                ]
            )
            and pd.notna(
                onset[column]
            )
        ):
            case[
                f"{label}_lead_days"
            ] = int(
                (
                    onset[
                        "dsci_normalized"
                    ]
                    - onset[column]
                ).days
            )
        else:
            case[
                f"{label}_lead_days"
            ] = None

        later_threshold = (
            bands[column][1]
        )

        earlier_threshold = (
            bands[column][0]
        )

        later = first_exceedance(
            holdout_weekly,
            column,
            later_threshold,
        )

        earlier = first_exceedance(
            holdout_weekly,
            column,
            earlier_threshold,
        )

        case[
            f"{label}_onset_later_threshold"
        ] = (
            str(later.date())
            if pd.notna(later)
            else None
        )

        case[
            f"{label}_onset_earlier_threshold"
        ] = (
            str(earlier.date())
            if pd.notna(earlier)
            else None
        )

    case[
        "lead_time_caveat"
    ] = (
        "Lead is measured against the USDM valid date, "
        "not publication time; it is not a claim of "
        "operational advance prediction."
    )

    #core-county onset uses thresholds matched inside the core-county set
    core_calibration_weekly = (
        calibration[
            calibration[
                "county_fips"
            ].isin(
                core_counties
            )
        ]
        .groupby(
            "release_date"
        )[
            [
                "dsci_normalized",
                "composite_prediction",
                "rf_prediction",
            ]
        ]
        .mean()
    )

    core_holdout_weekly = (
        holdout[
            holdout[
                "county_fips"
            ].isin(
                core_counties
            )
        ]
        .groupby(
            "release_date"
        )[
            [
                "dsci_normalized",
                "composite_prediction",
                "rf_prediction",
            ]
        ]
        .mean()
    )

    core_thresholds, _, _, _ = (
        matched_onset_thresholds(
            core_calibration_weekly
        )
    )

    for column, label in labels.items():
        core_onset = first_exceedance(
            core_holdout_weekly,
            column,
            core_thresholds[column],
        )

        case[
            f"core_{label}_onset"
        ] = (
            str(core_onset.date())
            if pd.notna(core_onset)
            else None
        )

    # Peak/severe weeks use regional mean USDM > 250.
    severe_weeks = (
        holdout_weekly.index[
            holdout_weekly[
                "dsci_normalized"
            ]
            > PEAK_DSCI
        ]
    )

    if len(severe_weeks):
        severe = holdout[
            holdout[
                "release_date"
            ].isin(
                severe_weeks
            )
        ]

        case[
            "peak_metric_basis"
        ] = "regional weeks with DSCI > 250"

    else:
        severe = holdout[
            holdout[
                "release_date"
            ]
            == PEAK_ALL_COUNTIES
        ]

        case[
            "peak_metric_basis"
        ] = "fixed peak-week fallback"

    case[
        "n_severe_weeks_regional"
    ] = len(
        severe_weeks
    )

    actual = severe[
        "dsci_normalized"
    ].to_numpy(dtype=float)

    for prediction, label in [
        (
            "composite_prediction",
            "composite",
        ),
        (
            "rf_prediction",
            "rf",
        ),
    ]:
        predicted = severe[
            prediction
        ].to_numpy(dtype=float)

        case[
            f"{label}_peak_rmse"
        ] = float(
            np.sqrt(
                np.mean(
                    (
                        predicted
                        - actual
                    ) ** 2
                )
            )
        )

        case[
            f"{label}_peak_mean_prediction"
        ] = float(
            predicted.mean()
        )

        case[
            f"{label}_peak_bias_signed"
        ] = float(
            (
                predicted
                - actual
            ).mean()
        )

    case["usdm_peak_mean"] = float(
        actual.mean()
    )

    calibration_rf_max = float(
        model.predict(
            calibration[
                features
            ]
        ).max()
    )

    severe_rf_max = float(
        severe[
            "rf_prediction"
        ].max()
    )

    case[
        "rf_ceiling_percent"
    ] = (
        severe_rf_max
        / calibration_rf_max
        * 100
        if calibration_rf_max > 0
        else np.nan
    )

    recovery = holdout_weekly.loc[
        RECOVERY_START:HOLDOUT_END
    ]

    for prediction, label in [
        (
            "composite_prediction",
            "composite",
        ),
        (
            "rf_prediction",
            "rf",
        ),
    ]:
        case[
            f"{label}_recovery_pearson"
        ] = float(
            np.corrcoef(
                recovery[
                    "dsci_normalized"
                ],
                recovery[
                    prediction
                ],
            )[0, 1]
        )

        case[
            f"{label}_recovery_bias"
        ] = float(
            (
                recovery[
                    prediction
                ]
                - recovery[
                    "dsci_normalized"
                ]
            ).mean()
        )

    path = CASE_FILE

    pd.DataFrame(
        [case]
    ).to_csv(
        path,
        index=False,
    )

    return path, core_counties


def main():
    features = feature_columns()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, weights, model = load_inputs()

    raw_weights = weights[
        "raw_weight"
    ].to_numpy(dtype=float)

    normalized_weights = weights[
        "normalized_weight"
    ].to_numpy(dtype=float)

    intercept, diagnostics = load_phase4_fit(
        features,
    )

    gate_phase4_reproduction(
        diagnostics,
        features,
        normalized_weights,
        raw_weights,
        intercept,
    )

    # Only rows with a target belong in calibration/holdout evaluation.
    data = data.dropna(
        subset=["dsci_normalized"]
    ).copy()

    data = add_predictions(
        data,
        features,
        normalized_weights,
        raw_weights,
        intercept,
        model,
    )

    calibration = data[
        (data["release_date"] >= CALIBRATION_START)
        & (data["release_date"] <= CALIBRATION_END)
    ].copy()

    holdout = data[
        (data["release_date"] >= HOLDOUT_START)
        & (data["release_date"] <= HOLDOUT_END)
    ].copy()

    holdout_weeks = (
        holdout[
            "release_date"
        ].nunique()
    )

    if holdout_weeks != EXPECTED_HOLDOUT_WEEKS:
        raise ValueError(
            f"Expected {EXPECTED_HOLDOUT_WEEKS} holdout weeks, "
            f"found {holdout_weeks}."
        )

 
    #overall and drought-conditional metrics
    metric_output = []

    metric_output += metric_rows(
        holdout,
        "composite_prediction",
        "composite",
    )

    metric_output += metric_rows(
        holdout,
        "rf_prediction",
        "rf",
    )

    #core counties are defined from the event window
    event = data[
        (data["release_date"] >= CORE_START)
        & (data["release_date"] <= CORE_END)
    ]

    core_counties = (
        event.groupby(
            "county_fips"
        )["DSCI"]
        .max()
    )

    core_counties = (
        core_counties[
            core_counties
            >= CORE_DSCI_MIN
        ]
        .index
        .tolist()
    )

    metric_output += anchor_rows(
        holdout,
        PEAK_ALL_COUNTIES,
        "PEAK_ALL_COUNTIES",
    )

    metric_output += anchor_rows(
        holdout,
        PEAK_CORE_COUNTIES,
        "PEAK_CORE_COUNTIES",
        county_fips=core_counties,
    )

    metrics_path = METRICS_FILE

    pd.DataFrame(
        metric_output
    ).to_csv(
        metrics_path,
        index=False,
    )

  
    #weekly regional series and matched onset thresholds
    series_columns = [
        "dsci_normalized",
        "composite_prediction",
        "rf_prediction",
    ]

    calibration_weekly = (
        calibration.groupby(
            "release_date"
        )[series_columns]
        .mean()
    )

    holdout_weekly = (
        holdout.groupby(
            "release_date"
        )[series_columns]
        .mean()
    )

    (
        thresholds,
        bands,
        onset_frequency,
        peak_frequency,
    ) = matched_onset_thresholds(
        calibration_weekly
    )

    threshold_rows = []

    for series in series_columns:
        threshold_rows.append(
            {
                "series": series,
                "onset_threshold": (
                    thresholds[series]
                ),
                "onset_threshold_lower": (
                    bands[series][0]
                ),
                "onset_threshold_upper": (
                    bands[series][1]
                ),
                "calibration_onset_frequency": (
                    onset_frequency
                ),
                "calibration_peak_frequency": (
                    peak_frequency
                ),
            }
        )

    thresholds_path = THRESHOLDS_FILE

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        thresholds_path,
        index=False,
    )

    weekly_path = WEEKLY_FILE

    holdout_weekly.reset_index().to_csv(
        weekly_path,
        index=False,
    )

    #2024 case study
    case_path, core_counties = (
        save_case_study(
            data,
            calibration,
            holdout,
            calibration_weekly,
            holdout_weekly,
            thresholds,
            bands,
            model,
            features,
        )
    )

 
    #seasonal evaluation
    seasonal_path = save_seasonal_metrics(
        holdout,
    )

    #RF holdout permutation importance
    permutation_table, permutation_path = (
        save_permutation_importance(
            model,
            holdout,
            features,
        )
    )

    comparison_path, top3_overlap, rho = compare_importance(
        weights,
        permutation_table,
        features,
    )

    
    #results
    print(
        f"Holdout: {len(holdout):,} rows across {holdout_weeks} weeks\n"
    )

    print(
        f"{'model':<10} {'stratum':<16} {'n':>7} "
        f"{'RMSE':>8} {'R2':>8} {'bias':>8}"
    )

    for row in metric_output:
        if row["stratum"] not in {"pooled", "conditional_gt0"}:
            continue

        print(
            f"{row['model']:<10} {row['stratum']:<16} {row['n']:>7,} "
            f"{row['rmse']:>8.4f} {row['r2']:>8.4f} "
            f"{row['bias_signed']:>8.4f}"
        )

    print(
        f"\nNNLS vs RF top-3 feature overlap: {top3_overlap}/3"
    )

    print(
        f"Spearman rank correlation:        {rho:.3f}"
    )

    print("\nDone. Output saved to:")

    for path in [
        metrics_path,
        case_path,
        seasonal_path,
        thresholds_path,
        weekly_path,
        permutation_path,
        comparison_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
