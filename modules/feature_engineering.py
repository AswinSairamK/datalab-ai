# ============================================================
# feature_engineering.py — Feature engineering module
# ============================================================
# Create new features from existing columns:
# 1. Mathematical transformations (log, sqrt, square, reciprocal)
# 2. Binning numeric columns
# 3. Polynomial features
# 4. Interaction features (multiply, add, subtract, divide)
# 5. Datetime feature extraction (year, month, day, weekday)
# 6. Text length features
# 7. Aggregations (groupby mean, sum, count)
# ============================================================

import pandas as pd
import numpy as np


def apply_math_transform(df: pd.DataFrame, column: str, transform: str) -> tuple:
    """
    Apply a mathematical transformation to a numeric column.
    
    Transforms:
    - "log": log(x+1) — handles zeros safely
    - "log10": log base 10
    - "sqrt": square root
    - "square": x²
    - "cube": x³
    - "reciprocal": 1/x
    - "exp": e^x
    - "abs": absolute value
    """
    df = df.copy()
    new_col = f"{column}_{transform}"
    
    try:
        if transform == "log":
            df[new_col] = np.log1p(df[column].abs())
        elif transform == "log10":
            df[new_col] = np.log10(df[column].abs() + 1)
        elif transform == "sqrt":
            df[new_col] = np.sqrt(df[column].abs())
        elif transform == "square":
            df[new_col] = df[column] ** 2
        elif transform == "cube":
            df[new_col] = df[column] ** 3
        elif transform == "reciprocal":
            df[new_col] = 1 / df[column].replace(0, np.nan)
        elif transform == "exp":
            # Clip to avoid overflow
            clipped = df[column].clip(upper=700)
            df[new_col] = np.exp(clipped)
        elif transform == "abs":
            df[new_col] = df[column].abs()
        else:
            return df, f"Unknown transform: {transform}"
        
        return df, f"Created '{new_col}' using {transform} transform"
    except Exception as e:
        return df, f"Error: {str(e)}"


def create_binned_feature(df: pd.DataFrame, column: str, n_bins: int = 5, method: str = "equal_width") -> tuple:
    """
    Create a binned version of a numeric column.
    
    Methods:
    - "equal_width": bins of equal width
    - "equal_frequency": bins with equal number of samples (quantiles)
    """
    df = df.copy()
    new_col = f"{column}_binned"
    
    try:
        if method == "equal_width":
            df[new_col] = pd.cut(df[column], bins=n_bins, labels=[f"Bin_{i+1}" for i in range(n_bins)])
        elif method == "equal_frequency":
            df[new_col] = pd.qcut(df[column], q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates='drop')
        else:
            return df, f"Unknown binning method: {method}"
        
        return df, f"Created '{new_col}' with {n_bins} bins ({method})"
    except Exception as e:
        return df, f"Error: {str(e)}"


def create_interaction(df: pd.DataFrame, col1: str, col2: str, operation: str) -> tuple:
    """
    Create an interaction feature between two columns.
    
    Operations:
    - "multiply": col1 * col2
    - "add": col1 + col2
    - "subtract": col1 - col2
    - "divide": col1 / col2
    - "ratio": col1 / (col1 + col2)
    """
    df = df.copy()
    
    try:
        if operation == "multiply":
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1] * df[col2]
        elif operation == "add":
            new_col = f"{col1}_plus_{col2}"
            df[new_col] = df[col1] + df[col2]
        elif operation == "subtract":
            new_col = f"{col1}_minus_{col2}"
            df[new_col] = df[col1] - df[col2]
        elif operation == "divide":
            new_col = f"{col1}_div_{col2}"
            df[new_col] = df[col1] / df[col2].replace(0, np.nan)
        elif operation == "ratio":
            new_col = f"{col1}_ratio"
            df[new_col] = df[col1] / (df[col1] + df[col2]).replace(0, np.nan)
        else:
            return df, f"Unknown operation: {operation}"
        
        return df, f"Created '{new_col}' using {col1} {operation} {col2}"
    except Exception as e:
        return df, f"Error: {str(e)}"


def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> tuple:
    """
    Create polynomial features (e.g., x², xy, y²).
    
    For degree=2 with columns [a, b], creates: a², a*b, b²
    """
    df = df.copy()
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Drop NaN for polynomial generation
        valid_mask = df[columns].notna().all(axis=1)
        
        if valid_mask.sum() == 0:
            return df, "No valid rows (all have NaN)"
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df.loc[valid_mask, columns])
        feature_names = poly.get_feature_names_out(columns)
        
        # Only keep the NEW features (skip original columns)
        new_features = [f for f in feature_names if f not in columns]
        new_indices = [i for i, f in enumerate(feature_names) if f in new_features]
        
        new_feature_df = pd.DataFrame(
            poly_features[:, new_indices],
            columns=new_features,
            index=df.loc[valid_mask].index
        )
        
        # Clean up column names (replace spaces with underscores)
        new_feature_df.columns = [c.replace(' ', '_').replace('^', '_pow_') for c in new_feature_df.columns]
        
        # Merge back into the main df
        for col in new_feature_df.columns:
            df[col] = np.nan
            df.loc[valid_mask, col] = new_feature_df[col].values
        
        return df, f"Created {len(new_feature_df.columns)} polynomial features (degree {degree})"
    except Exception as e:
        return df, f"Error: {str(e)}"


def extract_datetime_features(df: pd.DataFrame, column: str, features: list) -> tuple:
    """
    Extract features from a datetime column.
    
    Features:
    - "year", "month", "day", "hour", "minute"
    - "weekday" (0=Monday, 6=Sunday)
    - "week" (week of year)
    - "quarter"
    - "is_weekend"
    - "day_of_year"
    """
    df = df.copy()
    
    try:
        # Convert to datetime if not already
        dt_series = pd.to_datetime(df[column], errors='coerce')
        
        created = []
        
        if "year" in features:
            df[f"{column}_year"] = dt_series.dt.year
            created.append(f"{column}_year")
        if "month" in features:
            df[f"{column}_month"] = dt_series.dt.month
            created.append(f"{column}_month")
        if "day" in features:
            df[f"{column}_day"] = dt_series.dt.day
            created.append(f"{column}_day")
        if "hour" in features:
            df[f"{column}_hour"] = dt_series.dt.hour
            created.append(f"{column}_hour")
        if "minute" in features:
            df[f"{column}_minute"] = dt_series.dt.minute
            created.append(f"{column}_minute")
        if "weekday" in features:
            df[f"{column}_weekday"] = dt_series.dt.weekday
            created.append(f"{column}_weekday")
        if "week" in features:
            df[f"{column}_week"] = dt_series.dt.isocalendar().week
            created.append(f"{column}_week")
        if "quarter" in features:
            df[f"{column}_quarter"] = dt_series.dt.quarter
            created.append(f"{column}_quarter")
        if "is_weekend" in features:
            df[f"{column}_is_weekend"] = (dt_series.dt.weekday >= 5).astype(int)
            created.append(f"{column}_is_weekend")
        if "day_of_year" in features:
            df[f"{column}_day_of_year"] = dt_series.dt.dayofyear
            created.append(f"{column}_day_of_year")
        
        return df, f"Created {len(created)} datetime features: {', '.join(created)}"
    except Exception as e:
        return df, f"Error: {str(e)}"


def create_text_length_feature(df: pd.DataFrame, column: str) -> tuple:
    """Create a feature with the length of text values."""
    df = df.copy()
    new_col = f"{column}_length"
    
    try:
        df[new_col] = df[column].astype(str).str.len()
        df.loc[df[column].isna(), new_col] = np.nan
        return df, f"Created '{new_col}' with character counts"
    except Exception as e:
        return df, f"Error: {str(e)}"


def create_word_count_feature(df: pd.DataFrame, column: str) -> tuple:
    """Create a feature with word counts of text values."""
    df = df.copy()
    new_col = f"{column}_word_count"
    
    try:
        df[new_col] = df[column].astype(str).str.split().str.len()
        df.loc[df[column].isna(), new_col] = np.nan
        return df, f"Created '{new_col}' with word counts"
    except Exception as e:
        return df, f"Error: {str(e)}"