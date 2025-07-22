import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    RobustScaler,
)
import numpy as np


def preprocess_data(df, options: dict, target_column=None):
    # Column renaming
    if options.get("rename_columns"):
        df = df.rename(columns=options["rename_columns"])

    # Type conversion
    if options.get("convert_types"):
        for col, dtype in options["convert_types"].items():
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                pass  # Optionally log or handle conversion errors

    if options.get("columns_to_include"):
        df = df[options["columns_to_include"]]

    if options.get("drop_duplicates"):
        df = df.drop_duplicates()

    if options.get("dropna"):
        df = df.dropna()
    elif options.get("fillna"):
        method = options["fillna"]
        if method == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif method == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif method == "mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "zero":
            df = df.fillna(0)
        elif method == "custom":
            fill_value = options.get("fill_value", 0)
            df = df.fillna(fill_value)

    # Encoding
    if options.get("encode") == "label":
        for col in df.select_dtypes(include="object").columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif options.get("encode") == "onehot":
        df = pd.get_dummies(df)

    # Normalization
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not options.get("scale_target") and target_column:
        numeric_cols = [col for col in numeric_cols if col != target_column]

    if options.get("normalize") == "standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif options.get("normalize") == "minmax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif options.get("normalize") == "robust":
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Outlier removal
    if options.get("remove_outliers"):
        method = options.get("outlier_method", "zscore")
        numeric_cols = df.select_dtypes(include="number").columns
        if method == "zscore":
            from scipy.stats import zscore

            z_scores = df[numeric_cols].apply(zscore)
            df = df[(z_scores.abs() < 3).all(axis=1)]
        elif method == "iqr":
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~(
                (df[numeric_cols] < (Q1 - 1.5 * IQR))
                | (df[numeric_cols] > (Q3 + 1.5 * IQR))
            ).any(axis=1)
            df = df[mask]

    return df
