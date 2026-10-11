#Random Forest training
#train:      before 2016-01-01
#validation: 2016-01-01 through 2018-12-31
#holdout:    2019+ (NOT used in this script)

#quick test (a few minutes instead of hours):
#python scripts/phase5_rf.py --n-searches 5


import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_FILE = OUTPUT_DIR / "model_dataset_2000_2025.csv"

SEARCH_FILE = OUTPUT_DIR / "phase5_hyperparameter_search.csv"
AUDIT_FILE = OUTPUT_DIR / "phase5_stability_audit.csv"
MDI_FILE = OUTPUT_DIR / "phase5_mdi_importances.csv"
PROTOCOL_FILE = OUTPUT_DIR / "phase5_protocol_log.json"
MODEL_FILE = MODEL_DIR / "rf_droughtwatch_2000_2018.joblib"

TRAIN_END = pd.Timestamp("2016-01-01")
VALID_END = pd.Timestamp("2019-01-01")
HOLDOUT_END = pd.Timestamp("2025-03-25")

DEFAULT_SEARCHES = 100
RANDOM_SEED = 42
AUDIT_SPLITS = 5

PARAMETERS = {
    "n_estimators": randint(100, 1000),
    "max_depth": [None] + list(range(5, 51)),
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.3, 0.4),
}


def feature_columns():
    #same eight features used by NNLS
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


def rmse(actual, predicted):
   #root mean squared error
    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def load_dataset():
    #only rows with an observed DSCI target are kept for model fitting
    path = DATASET_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run calibrate_weights.py first."
        )

    data = pd.read_csv(
        path,
        parse_dates=["release_date"],
        dtype={"county_fips": str},
    )

    features = feature_columns()

    required = features + ["dsci_normalized"]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Model dataset is missing columns: {missing}"
        )

    # hase 5 fits only rows that have a USDM target.
    data = data.dropna(
        subset=["dsci_normalized"]
    ).copy()

    feature_null_rows = data[features].isna().any(axis=1)

    if feature_null_rows.any():
        raise ValueError(
            f"{int(feature_null_rows.sum()):,} rows with DSCI "
            "have missing RF features. Fix the model dataset first."
        )

    return data


def split_dataset(data):
    train = data[
        data["release_date"] < TRAIN_END
    ].copy()

    validation = data[
        (data["release_date"] >= TRAIN_END)
        & (data["release_date"] < VALID_END)
    ].copy()

    holdout = data[
        (data["release_date"] >= VALID_END)
        & (data["release_date"] <= HOLDOUT_END)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Train or validation split is empty."
        )

    return train, validation, holdout


def clean_parameters(parameters):
    #convert scipy/numpy parameter values into normal Python values
    cleaned = dict(parameters)

    cleaned["n_estimators"] = int(
        cleaned["n_estimators"]
    )

    cleaned["min_samples_leaf"] = int(
        cleaned["min_samples_leaf"]
    )

    if cleaned["max_depth"] is None or pd.isna(cleaned["max_depth"]):
        cleaned["max_depth"] = None
    else:
        cleaned["max_depth"] = int(
            cleaned["max_depth"]
        )

    cleaned["max_features"] = float(
        cleaned["max_features"]
    )

    return cleaned


def search_parameters(
    train,
    validation,
    features,
    n_searches,
):
    
    #try random RF settings
    #Every candidate is: fit on 2000-2015 and scored on 2016-2018
    #the lowest validation RMSE wins automatically
    X_train = train[features]
    y_train = train["dsci_normalized"].to_numpy()

    X_valid = validation[features]
    y_valid = validation["dsci_normalized"].to_numpy()

    candidates = ParameterSampler(
        PARAMETERS,
        n_iter=n_searches,
        random_state=RANDOM_SEED,
    )

    rows = []

    for number, parameters in enumerate(
        candidates,
        start=1,
    ):
        parameters = clean_parameters(
            parameters
        )

        model = RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **parameters,
        )

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_valid
        )

        score = rmse(
            y_valid,
            prediction,
        )

        rows.append(
            {
                **parameters,
                "validation_rmse": score,
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )

    return results


def save_search_results(
    results,
):
    #save all tested parameter combinations
    path = SEARCH_FILE

    results.to_csv(
        path,
        index=False,
    )

    return path


def best_parameters(results):
    #return the row with the lowest validation RMSE
    row = results.iloc[0].to_dict()
    row.pop("validation_rmse")

    return clean_parameters(row)


def fit_final_model(
    data,
    features,
    parameters,
):
    #refit the winning RF on all pre-holdout data: 2000-2018
    fit_data = data[
        data["release_date"] < VALID_END
    ]

    model = RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **parameters,
    )

    model.fit(
        fit_data[features],
        fit_data["dsci_normalized"].to_numpy(),
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = MODEL_FILE

    joblib.dump(
        model,
        path,
    )

    return model, path


def run_stability_audit(
    train,
    features,
    parameters,
):
    
    #refit the already-selected RF settings across five expanding time folds
    weeks = np.array(
        sorted(
            train["release_date"].unique()
        )
    )

    splitter = TimeSeriesSplit(
        n_splits=AUDIT_SPLITS,
        gap=1,
    )

    rows = []

    for fold, (
        train_week_index,
        validation_week_index,
    ) in enumerate(
        splitter.split(weeks),
        start=1,
    ):
        train_weeks = weeks[
            train_week_index
        ]

        validation_weeks = weeks[
            validation_week_index
        ]

        fold_train = train[
            train["release_date"].isin(
                train_weeks
            )
        ]

        fold_valid = train[
            train["release_date"].isin(
                validation_weeks
            )
        ]

        model = RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **parameters,
        )

        model.fit(
            fold_train[features],
            fold_train["dsci_normalized"].to_numpy(),
        )

        actual = (
            fold_valid["dsci_normalized"]
            .to_numpy()
        )

        predicted = model.predict(
            fold_valid[features]
        )

        correlation = (
            np.corrcoef(
                actual,
                predicted,
            )[0, 1]
            if len(actual) > 1
            else np.nan
        )

        rows.append(
            {
                "fold": fold,
                "validation_start": (
                    fold_valid["release_date"]
                    .min()
                    .date()
                ),
                "validation_end": (
                    fold_valid["release_date"]
                    .max()
                    .date()
                ),
                "train_weeks": len(
                    train_weeks
                ),
                "validation_weeks": len(
                    validation_weeks
                ),
                "validation_dsci_mean": float(
                    actual.mean()
                ),
                "validation_fraction_positive": float(
                    (actual > 0).mean()
                ),
                "validation_rmse": rmse(
                    actual,
                    predicted,
                ),
                "validation_pearson": float(
                    correlation
                ),
            }
        )

    audit = pd.DataFrame(rows)

    path = AUDIT_FILE

    audit.to_csv(
        path,
        index=False,
    )

    return path


def save_mdi_importance(
    model,
    features,
):
    #save Random Forest mean-decrease-in-impurity importance
    #description of how much/how strongly the final forest used each feature
    importance = pd.DataFrame(
        {
            "feature": features,
            "mdi_importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "mdi_importance",
        ascending=False,
    )

    path = MDI_FILE

    importance.to_csv(
        path,
        index=False,
    )

    return importance, path


def save_protocol(
    parameters,
    n_searches,
):
    #save the frozen modeling decisions used for this run
    protocol = {
        "features": feature_columns(),
        "train_end": str(
            TRAIN_END.date()
        ),
        "validation_end_holdout_start": str(
            VALID_END.date()
        ),
        "holdout_end": str(
            HOLDOUT_END.date()
        ),
        "random_search_candidates": int(
            n_searches
        ),
        "random_seed": RANDOM_SEED,
        "selection_rule": (
            "lowest 2016-2018 validation RMSE"
        ),
        "refit_rule": (
            "winning settings refit on all 2000-2018 rows"
        ),
        "holdout_rule": (
            "2019+ not used by Phase 5; evaluate in Phase 6"
        ),
        "selected_parameters": parameters,
    }

    path = PROTOCOL_FILE

    path.write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Random Forest training"
        )
    )

    parser.add_argument(
        "--n-searches",
        type=int,
        default=DEFAULT_SEARCHES,
        help=(
            "Number of random hyperparameter settings. "
        ),
    )

    args = parser.parse_args()

    if args.n_searches < 1:
        raise ValueError(
            "--n-searches must be at least 1"
        )

    features = feature_columns()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_dataset()

    #holdout is split off here only so it is never touched below
    train, validation, holdout = (
        split_dataset(data)
    )

    results = search_parameters(
        train,
        validation,
        features,
        args.n_searches,
    )

    search_path = save_search_results(
        results,
    )

    selected = best_parameters(
        results
    )

    model, model_path = fit_final_model(
        data,
        features,
        selected,
    )

    audit_path = run_stability_audit(
        train,
        features,
        selected,
    )

    importance, mdi_path = save_mdi_importance(
        model,
        features,
    )

    protocol_path = save_protocol(
        selected,
        args.n_searches,
    )

    print(
        f"Candidates tried:     {args.n_searches}"
    )

    print(
        f"Best validation RMSE: "
        f"{results['validation_rmse'].iloc[0]:.6f}"
    )

    print(
        f"Selected settings:    {selected}"
    )

    print("\nFeature importance (MDI):")

    for row in importance.itertuples():
        print(
            f"  {row.feature:<22} {row.mdi_importance:.4f}"
        )

    print("\nDone. Output saved to:")

    for path in [
        search_path,
        audit_path,
        mdi_path,
        protocol_path,
        model_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
