from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DATA_DIR, MODEL_DIR
from .facilities import FACILITY_TYPES


MIN_TRAINING_ROWS = 25
MODEL_ARTIFACT_VERSION = 2


def _ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def train_model_from_dataset(dataset: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Train a valuation model only from an observed, positive price target.

    BAG characteristics are useful features, but they are not a price source.  In
    particular, this function deliberately refuses to manufacture a target from
    living area or construction year: doing so produces impressive but meaningless
    validation scores.
    """
    if dataset.empty:
        raise ValueError("Dataset is leeg; geen modeltraining mogelijk.")
    if target_column not in dataset.columns:
        raise ValueError(f"Prijsdoel '{target_column}' ontbreekt in de dataset.")

    dataset = dataset.copy()

    numeric_columns = [
        "oppervlakte_m2",
        "bouwjaar",
        "age_years",
        "log_oppervlakte",
        "aantal_verblijfsobjecten",
        "is_apartment",
        "area_per_unit",
        "huisnummer",
        "postcode4_num",
        "latitude",
        "longitude",
        "status_is_active",
        "oppervlakte_bekend",
        "bouwjaar_bekend",
        "oppervlakte_m2_funda",
        "overige_inpandige_ruimte_m2",
        "externe_bergruimte_m2",
        "gebouwgebonden_buitenruimte_m2",
        "perceeloppervlakte_m2",
        "inhoud_m3",
        "vraagprijs_per_m2",
        "bouwjaar_funda",
        "kamers",
        "slaapkamers",
        "badkamers",
        "woonlagen",
        "oppervlakteverschil_funda_bag",
        "perceel_beschikbaar",
        "energielabel_score",
        "bron_bag_bekend",
        "bron_cbs_beschikbaar",
        "bron_cbs_records",
        "bron_cbs_inkomen_beschikbaar",
        "bron_energielabel_beschikbaar",
        "bron_osm_beschikbaar",
        "bron_bgt_beschikbaar",
        "bron_brt_beschikbaar",
    ] + [f"afstand_{kind}_km" for kind in FACILITY_TYPES]
    dataset[target_column] = pd.to_numeric(dataset[target_column], errors="coerce")
    dataset = dataset[dataset[target_column].gt(0)].copy()
    if len(dataset) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Minimaal {MIN_TRAINING_ROWS} geldige, positieve prijsrecords nodig "
            f"voor modeltraining ({target_column}); gevonden: {len(dataset)}."
        )

    defaults = {
        "oppervlakte_m2": 100.0,
        "bouwjaar": 1980.0,
        "age_years": 46.0,
        "log_oppervlakte": np.log1p(100.0),
        "aantal_verblijfsobjecten": 1.0,
        "is_apartment": 0.0,
        "area_per_unit": 100.0,
    }
    for column in numeric_columns + [target_column]:
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
            median_value = dataset[column].median()
            fill_value = median_value if pd.notna(median_value) else defaults.get(column, 0.0)
            dataset[column] = dataset[column].fillna(fill_value)

    feature_columns = [
        "oppervlakte_m2",
        "bouwjaar",
        "age_years",
        "log_oppervlakte",
        "aantal_verblijfsobjecten",
        "is_apartment",
        "area_per_unit",
        "huisnummer",
        "postcode4_num",
        "latitude",
        "longitude",
        "status_is_active",
        "oppervlakte_bekend",
        "bouwjaar_bekend",
        "provincie",
        "woonplaats",
        "gebruiksdoel",
        "status",
        "pand_status",
        "pand_gebruiksdoel",
        "postcode4",
        "energielabel",
        "woningtype",
        "soort_bouw",
        "soort_dak",
        "isolatie",
        "verwarming",
        "warm_water",
        "tuin",
        "ligging_tuin",
        "parkeergelegenheid",
        "energielabel_score",
    ] + [f"afstand_{kind}_km" for kind in FACILITY_TYPES]
    feature_columns = [
        column
        for column in feature_columns
        if column in dataset.columns and not dataset[column].isna().all()
    ]
    # Keep every usable feature. Numeric gaps use the training median through
    # the pipeline; text gaps get an explicit category instead of disappearing.
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(dataset[column]):
            dataset[column] = dataset[column].fillna("onbekend").astype(str)

    X = dataset[feature_columns].copy()
    y = dataset[target_column]

    categorical_features = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(X[column])]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }

    test_count = max(5, round(len(dataset) * 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=42
    )
    results: List[Dict[str, Any]] = []
    best_model_name = None
    best_score = -np.inf
    best_residuals: np.ndarray | None = None

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "model": name,
            "r2": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        }
        results.append(metrics)
        if np.isfinite(metrics["r2"]) and metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model_name = name
            best_residuals = np.abs(y_test.to_numpy() - predictions)

    if best_model_name is None or best_residuals is None:
        raise ValueError("Modelvalidatie leverde geen geldige R²-score op.")

    model = models[best_model_name]
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    best_pipeline.fit(X, y)
    interval_half_width = float(np.quantile(best_residuals, 0.9))
    artifact_path = _ensure_model_dir() / "best_housing_model.pkl"
    temporary_path = artifact_path.with_suffix(".tmp")
    artifact = {
        "version": MODEL_ARTIFACT_VERSION,
        "pipeline": best_pipeline,
        "metadata": {
            "target_column": target_column,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(len(dataset)),
            "validation_rows": int(len(y_test)),
            "interval_half_width": interval_half_width,
            "best_score": float(best_score),
        },
    }
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(artifact_path)

    return {
        "results": results,
        "best_model": best_model_name,
        "best_score": float(best_score),
        "artifact": str(artifact_path),
        "training_rows": int(len(dataset)),
        "validation_rows": int(len(y_test)),
        "interval_half_width": interval_half_width,
    }


def is_current_model_artifact(model_artifact: str | Path) -> bool:
    """Return whether a saved artifact can be used for predictions.

    Older releases stored the Pipeline directly, while current releases store
    it together with version and prediction metadata.
    """
    try:
        artifact = joblib.load(model_artifact)
    except (OSError, ValueError, EOFError, ImportError, AttributeError):
        return False
    if isinstance(artifact, Pipeline):
        return True
    return (
        isinstance(artifact, dict)
        and artifact.get("version") == MODEL_ARTIFACT_VERSION
        and isinstance(artifact.get("pipeline"), Pipeline)
    )


def model_input_features(model_artifact: str | Path) -> list[str]:
    """Return the columns learned by a saved pipeline."""
    artifact = joblib.load(model_artifact)
    model = artifact if isinstance(artifact, Pipeline) else artifact.get("pipeline")
    if not isinstance(model, Pipeline):
        return []
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        return []
    columns: list[str] = []
    for _, _, transformer_columns in preprocessor.transformers_:
        columns.extend(str(column) for column in transformer_columns)
    return columns


def predict_house_value(model_artifact: str, row: pd.Series) -> Dict[str, Any]:
    artifact_path = Path(model_artifact).resolve()
    model_dir = MODEL_DIR.resolve()
    if model_dir not in artifact_path.parents:
        raise ValueError("Modelbestand moet in de geconfigureerde modelmap staan.")
    if not artifact_path.is_file():
        raise FileNotFoundError("Modelbestand niet gevonden.")

    artifact = joblib.load(artifact_path)
    if isinstance(artifact, Pipeline):
        model = artifact
        metadata: dict[str, Any] = {}
    elif isinstance(artifact, dict) and artifact.get("version") == MODEL_ARTIFACT_VERSION:
        model = artifact.get("pipeline")
        metadata = artifact.get("metadata", {})
    else:
        raise ValueError("Het modelbestand heeft geen bruikbaar ondersteund formaat.")
    if not isinstance(model, Pipeline):
        raise ValueError("Modelbestand bevat geen geldig model.")
    frame = pd.DataFrame([row])
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is not None:
        required_columns = []
        numeric_columns = set()
        categorical_columns = set()
        for name, _, columns in preprocessor.transformers_:
            normalized_columns = [str(column) for column in columns]
            required_columns.extend(normalized_columns)
            if name == "numeric":
                numeric_columns.update(normalized_columns)
            elif name == "categorical":
                categorical_columns.update(normalized_columns)
        for column in required_columns:
            if column not in frame.columns:
                frame[column] = np.nan if column in numeric_columns else "onbekend"
        for column in numeric_columns.intersection(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in categorical_columns.intersection(frame.columns):
            frame[column] = frame[column].fillna("onbekend").astype(str)
        frame = frame[required_columns]
    prediction = float(model.predict(frame)[0])
    interval_half_width = float(metadata.get("interval_half_width", 0.0))
    lower = max(0.0, prediction - interval_half_width)
    upper = prediction + interval_half_width
    return {"prediction": prediction, "lower_bound": lower, "upper_bound": upper}
