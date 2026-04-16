# ============================================================
# preprocessing.py — Data preprocessing module
# ============================================================
# Handles:
# 1. Missing values (drop, fill with mean/median/mode/custom)
# 2. Duplicate removal
# 3. Categorical encoding (label, one-hot, ordinal)
# 4. Numeric scaling (standard, min-max, robust)
# 5. Column removal
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)


def handle_missing_values(df: pd.DataFrame, column: str, method: str, fill_value=None) -> pd.DataFrame:
    """
    Handle missing values in a column.
    
    Methods:
    - "drop": drop rows with missing values in this column
    - "mean": fill with mean (numeric only)
    - "median": fill with median (numeric only)
    - "mode": fill with most common value
    - "forward_fill": use previous value
    - "backward_fill": use next value
    - "custom": fill with a specific value
    - "zero": fill with 0
    """
    df = df.copy()
    
    if method == "drop":
        df = df.dropna(subset=[column])
    
    elif method == "mean":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
    
    elif method == "median":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
    
    elif method == "mode":
        mode_val = df[column].mode()
        if len(mode_val) > 0:
            df[column] = df[column].fillna(mode_val[0])
    
    elif method == "forward_fill":
        df[column] = df[column].ffill()
    
    elif method == "backward_fill":
        df[column] = df[column].bfill()
    
    elif method == "custom":
        df[column] = df[column].fillna(fill_value)
    
    elif method == "zero":
        df[column] = df[column].fillna(0)
    
    return df


def drop_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def encode_categorical(df: pd.DataFrame, column: str, method: str) -> tuple:
    """
    Encode a categorical column.
    
    Methods:
    - "label": convert to integers (0, 1, 2, ...)
    - "one_hot": create binary columns for each category
    - "ordinal": similar to label but preserves order if specified
    
    Returns:
        (new_df, info_message)
    """
    df = df.copy()
    
    if method == "label":
        encoder = LabelEncoder()
        # Fill NaN with a placeholder, encode, then restore NaN
        non_null_mask = df[column].notna()
        if non_null_mask.any():
            encoded = encoder.fit_transform(df.loc[non_null_mask, column].astype(str))
            # Create a new numeric column (avoid dtype conflict)
            new_values = pd.Series(np.nan, index=df.index, dtype='float64')
            new_values.loc[non_null_mask] = encoded
            df[column] = new_values
            # Convert to Int64 (nullable integer) for cleaner display
            df[column] = df[column].astype('Int64')
        info = f"Label encoded '{column}' — {len(encoder.classes_)} unique categories mapped to integers"
    
    elif method == "one_hot":
        dummies = pd.get_dummies(df[column], prefix=column, dummy_na=False)
        # Convert bool to int for ML compatibility
        dummies = dummies.astype(int)
        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)
        info = f"One-hot encoded '{column}' — created {dummies.shape[1]} new binary columns"
    
    elif method == "ordinal":
        encoder = OrdinalEncoder()
        non_null_mask = df[column].notna()
        if non_null_mask.any():
            values = df.loc[non_null_mask, column].astype(str).values.reshape(-1, 1)
            encoded = encoder.fit_transform(values).flatten()
            new_values = pd.Series(np.nan, index=df.index, dtype='float64')
            new_values.loc[non_null_mask] = encoded
            df[column] = new_values
            df[column] = df[column].astype('Int64')
        info = f"Ordinal encoded '{column}'"
    
    return df, info


def scale_features(df: pd.DataFrame, columns: list, method: str) -> tuple:
    """
    Scale numeric columns.
    
    Methods:
    - "standard": zero mean, unit variance (StandardScaler)
    - "minmax": scale to [0, 1] range
    - "robust": uses median and IQR (robust to outliers)
    
    Returns:
        (new_df, info_message)
    """
    df = df.copy()
    
    if not columns:
        return df, "No columns selected"
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return df, f"Unknown method: {method}"
    
    # Drop rows with NaN in selected columns before scaling
    valid_mask = df[columns].notna().all(axis=1)
    
    if valid_mask.sum() == 0:
        return df, "No valid rows to scale (all have NaN)"
    
    df.loc[valid_mask, columns] = scaler.fit_transform(df.loc[valid_mask, columns])
    
    info = f"Scaled {len(columns)} columns using {method} scaling on {valid_mask.sum()} rows"
    return df, info


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop specified columns."""
    return df.drop(columns=columns, errors='ignore')


def convert_dtype(df: pd.DataFrame, column: str, new_type: str) -> tuple:
    """
    Convert column data type.
    
    Types: int, float, str, category, datetime
    """
    df = df.copy()
    
    try:
        if new_type == "int":
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif new_type == "float":
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == "str":
            df[column] = df[column].astype(str)
        elif new_type == "category":
            df[column] = df[column].astype('category')
        elif new_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors='coerce')
        else:
            return df, f"Unknown type: {new_type}"
        
        return df, f"Converted '{column}' to {new_type}"
    except Exception as e:
        return df, f"Error: {str(e)}"


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get a summary of missing values per column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
        "Total Rows": len(df)
    })
    
    summary = summary[summary["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    return summary