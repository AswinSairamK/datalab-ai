# ============================================================
# outliers.py — Outlier detection module
# ============================================================
# Implements multiple outlier detection methods:
# 1. IQR method (Interquartile Range)
# 2. Z-score method
# 3. Modified Z-score (using median)
# 4. Isolation Forest (ML-based)
# 5. Local Outlier Factor (LOF)
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> dict:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    
    Returns:
        - outlier_mask: boolean Series (True = outlier)
        - lower_bound, upper_bound: thresholds
        - count, percentage: outlier counts
    """
    data = series.dropna()
    
    if len(data) < 4:
        return {"error": "Not enough data points"}
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    
    return {
        "method": "IQR",
        "Q1": round(Q1, 3),
        "Q3": round(Q3, 3),
        "IQR": round(IQR, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_count / len(data) * 100, 2),
        "outlier_mask": outlier_mask,
        "outlier_values": series[outlier_mask].tolist()[:50],
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """
    Detect outliers using the Z-score method.
    
    Outliers are values where |z-score| > threshold (default 3).
    Z-score = (value - mean) / std
    """
    data = series.dropna()
    
    if len(data) < 4:
        return {"error": "Not enough data points"}
    
    mean = data.mean()
    std = data.std()
    
    if std == 0:
        return {"error": "Standard deviation is 0 (all values identical)"}
    
    z_scores = np.abs((series - mean) / std)
    outlier_mask = z_scores > threshold
    outlier_count = int(outlier_mask.sum())
    
    return {
        "method": "Z-score",
        "mean": round(mean, 3),
        "std": round(std, 3),
        "threshold": threshold,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_count / len(data) * 100, 2),
        "outlier_mask": outlier_mask,
        "outlier_values": series[outlier_mask].tolist()[:50],
    }


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> dict:
    """
    Detect outliers using the Modified Z-score method (uses median).
    
    More robust than regular Z-score because median is not affected
    by extreme outliers like the mean is.
    
    Modified Z-score = 0.6745 * (value - median) / MAD
    Where MAD = Median Absolute Deviation
    """
    data = series.dropna()
    
    if len(data) < 4:
        return {"error": "Not enough data points"}
    
    median = data.median()
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return {"error": "Median absolute deviation is 0"}
    
    modified_z = 0.6745 * (series - median) / mad
    outlier_mask = np.abs(modified_z) > threshold
    outlier_count = int(outlier_mask.sum())
    
    return {
        "method": "Modified Z-score",
        "median": round(median, 3),
        "MAD": round(mad, 3),
        "threshold": threshold,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_count / len(data) * 100, 2),
        "outlier_mask": outlier_mask,
        "outlier_values": series[outlier_mask].tolist()[:50],
    }


def detect_outliers_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.1) -> dict:
    """
    Detect outliers using Isolation Forest (unsupervised ML).
    
    Works on multiple columns at once — finds rows that are
    "isolated" from the rest of the data.
    
    contamination: expected proportion of outliers (0.0 to 0.5)
    """
    from sklearn.ensemble import IsolationForest
    
    if not columns:
        return {"error": "No columns selected"}
    
    # Get numeric data and drop NaN rows
    data = df[columns].dropna()
    
    if len(data) < 10:
        return {"error": "Not enough data points (need at least 10)"}
    
    # Fit and predict
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = iso.fit_predict(data)
    
    # -1 = outlier, 1 = normal
    outlier_indices = data.index[predictions == -1]
    outlier_count = len(outlier_indices)
    
    # Create boolean mask for the original DataFrame
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[outlier_indices] = True
    
    # Get anomaly scores (lower = more anomalous)
    scores = iso.score_samples(data)
    
    return {
        "method": "Isolation Forest",
        "contamination": contamination,
        "columns_used": columns,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_count / len(data) * 100, 2),
        "outlier_mask": outlier_mask,
        "score_min": round(scores.min(), 3),
        "score_max": round(scores.max(), 3),
        "score_mean": round(scores.mean(), 3),
    }


def detect_outliers_lof(df: pd.DataFrame, columns: list, n_neighbors: int = 20, contamination: float = 0.1) -> dict:
    """
    Detect outliers using Local Outlier Factor (LOF).
    
    LOF finds outliers based on local density — points that
    have substantially lower density than their neighbors.
    
    Better than Isolation Forest for finding local outliers
    in clustered data.
    """
    from sklearn.neighbors import LocalOutlierFactor
    
    if not columns:
        return {"error": "No columns selected"}
    
    data = df[columns].dropna()
    
    if len(data) < n_neighbors + 1:
        return {"error": f"Not enough data points (need at least {n_neighbors + 1})"}
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(data)
    
    outlier_indices = data.index[predictions == -1]
    outlier_count = len(outlier_indices)
    
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[outlier_indices] = True
    
    return {
        "method": "Local Outlier Factor",
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "columns_used": columns,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_count / len(data) * 100, 2),
        "outlier_mask": outlier_mask,
    }


def compare_outlier_methods(series: pd.Series) -> pd.DataFrame:
    """
    Run all single-column outlier methods and compare results.
    """
    results = []
    
    iqr_result = detect_outliers_iqr(series)
    if "error" not in iqr_result:
        results.append({
            "Method": "IQR (1.5x)",
            "Outliers Found": iqr_result["outlier_count"],
            "Percentage": f"{iqr_result['outlier_percentage']}%",
            "Description": "Most common method, detects extreme values"
        })
    
    zscore_result = detect_outliers_zscore(series)
    if "error" not in zscore_result:
        results.append({
            "Method": "Z-score (3σ)",
            "Outliers Found": zscore_result["outlier_count"],
            "Percentage": f"{zscore_result['outlier_percentage']}%",
            "Description": "Assumes normal distribution"
        })
    
    mod_z_result = detect_outliers_modified_zscore(series)
    if "error" not in mod_z_result:
        results.append({
            "Method": "Modified Z-score (3.5)",
            "Outliers Found": mod_z_result["outlier_count"],
            "Percentage": f"{mod_z_result['outlier_percentage']}%",
            "Description": "Robust to extreme values"
        })
    
    return pd.DataFrame(results)