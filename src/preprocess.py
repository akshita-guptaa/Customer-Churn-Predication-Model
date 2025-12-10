
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_telco_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # empty strings -> convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Ensure tenure and MonthlyCharges are numeric as well
    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_features_and_target(df: pd.DataFrame):
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    # Map Yes/No to 1/0
    y = y_raw.map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError(
            "Target column contains values other than 'Yes'/'No'. "
            "Check the dataset."
        )

    return X, y


def split_feature_types(X: pd.DataFrame):
    
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    if ID_COLUMN in categorical_features:
        categorical_features.remove(ID_COLUMN)
    if ID_COLUMN in numeric_features:
        numeric_features.remove(ID_COLUMN)

    return categorical_features, numeric_features


def build_preprocessor(categorical_features, numeric_features):
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    return preprocessor
