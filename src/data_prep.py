"""
MedRisk - data loading and preprocessing.

Dataset: UCI Heart Disease, Cleveland subset (303 patients, 13 clinical
attributes + diagnosis). This is the same file used in the UCI ML Repository
entry `heart+disease`, with the categorical codes already expanded to labels.

The two columns with genuine missing values in the original UCI file (`Ca`
and `Thal`, encoded as "?" upstream) are preserved as NaN here, so the
imputation step below is doing real work rather than a demonstration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_cleveland.csv"

TARGET = "AHD"

# Continuous / ordered-count measurements -> median impute + standardise.
NUMERIC = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak", "Ca"]

# Unordered clinical categories -> mode impute + one-hot.
CATEGORICAL = ["ChestPain", "Thal", "RestECG", "Slope"]

# Already 0/1. Scaling a binary flag only obscures the coefficient, so these
# pass through untouched.
BINARY = ["Sex", "Fbs", "ExAng"]

FEATURES = NUMERIC + CATEGORICAL + BINARY

COLUMN_GLOSSARY = {
    "Age": "Age in years",
    "Sex": "1 = male, 0 = female",
    "ChestPain": "Chest pain type (typical, atypical, non-anginal, asymptomatic)",
    "RestBP": "Resting blood pressure on admission (mm Hg)",
    "Chol": "Serum cholesterol (mg/dl)",
    "Fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "RestECG": "Resting electrocardiographic result (0, 1, 2)",
    "MaxHR": "Maximum heart rate achieved during stress test",
    "ExAng": "Exercise-induced angina (1 = yes)",
    "Oldpeak": "ST depression induced by exercise relative to rest",
    "Slope": "Slope of the peak exercise ST segment (1, 2, 3)",
    "Ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "Thal": "Thalassemia stress-test result (normal, fixed, reversable)",
    "AHD": "Angiographic heart disease present (target)",
}


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and drop the unnamed row-index column shipped with it."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df


def load_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature frame and a 0/1 target vector."""
    df = load_raw(path)

    # RestECG and Slope arrive as integers but are nominal/ordinal codes, not
    # magnitudes. Cast to string so the one-hot encoder treats them correctly.
    for col in ("RestECG", "Slope"):
        df[col] = df[col].astype("Int64").astype(str).replace("<NA>", np.nan)

    y = (df[TARGET] == "Yes").astype(int)
    X = df[FEATURES].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Impute -> scale/encode. Fitted inside each model pipeline so that the
    imputer and scaler only ever see training-fold statistics."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )


def feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Readable names for the columns that come out of the preprocessor."""
    return [n.split("__", 1)[-1] for n in fitted_preprocessor.get_feature_names_out()]


def missingness_report(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = load_raw(path)
    n_missing = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "missing": n_missing,
                "pct": (100 * n_missing / len(df)).round(2),
                "dtype": df.dtypes.astype(str),
            }
        )
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )
