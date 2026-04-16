# ============================================================
# eda.py — Exploratory Data Analysis module
# ============================================================
# Provides descriptive statistics, distributions, and
# correlation analysis for numeric columns.
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns.
    
    Returns a DataFrame with:
    - count, mean, median, std, variance
    - min, max, range
    - 25%, 50%, 75% quartiles
    - skewness, kurtosis
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats_data = []
    
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        stats_data.append({
            "Column": col,
            "Count": len(col_data),
            "Mean": round(col_data.mean(), 3),
            "Median": round(col_data.median(), 3),
            "Std": round(col_data.std(), 3),
            "Variance": round(col_data.var(), 3),
            "Min": round(col_data.min(), 3),
            "Max": round(col_data.max(), 3),
            "Range": round(col_data.max() - col_data.min(), 3),
            "Q1 (25%)": round(col_data.quantile(0.25), 3),
            "Q2 (50%)": round(col_data.quantile(0.50), 3),
            "Q3 (75%)": round(col_data.quantile(0.75), 3),
            "IQR": round(col_data.quantile(0.75) - col_data.quantile(0.25), 3),
            "Skewness": round(col_data.skew(), 3),
            "Kurtosis": round(col_data.kurtosis(), 3),
        })
    
    return pd.DataFrame(stats_data)


def get_categorical_stats(df: pd.DataFrame) -> dict:
    """
    Get statistics for categorical columns.
    
    Returns a dict where key is column name and value is
    a DataFrame showing value counts and percentages.
    """
    cat_df = df.select_dtypes(include=['object', 'category', 'bool'])
    
    if cat_df.empty:
        return {}
    
    result = {}
    
    for col in cat_df.columns:
        value_counts = df[col].value_counts()
        total = len(df[col].dropna())
        
        summary = pd.DataFrame({
            "Value": value_counts.index,
            "Count": value_counts.values,
            "Percentage": [f"{round(v/total*100, 2)}%" for v in value_counts.values]
        })
        
        # Only keep top 20 values for large categorical columns
        result[col] = summary.head(20)
    
    return result


def get_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    
    Methods:
    - "pearson": linear correlation (default)
    - "spearman": rank correlation (for non-linear relationships)
    - "kendall": Kendall's tau
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method)


def get_top_correlations(corr_matrix: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Get the top N highest correlations (excluding self-correlations).
    
    Returns a DataFrame with:
    - Feature 1
    - Feature 2
    - Correlation
    - Strength (weak/moderate/strong)
    """
    if corr_matrix.empty:
        return pd.DataFrame()
    
    # Get upper triangle to avoid duplicates
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Stack and sort
    corr_pairs = upper.stack().reset_index()
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    
    # Sort by absolute value (strongest first, positive or negative)
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    corr_pairs = corr_pairs.sort_values("Abs Correlation", ascending=False)
    
    # Add strength category
    def strength(val):
        abs_val = abs(val)
        if abs_val >= 0.7:
            return "Strong"
        elif abs_val >= 0.4:
            return "Moderate"
        elif abs_val >= 0.2:
            return "Weak"
        else:
            return "Very weak"
    
    corr_pairs["Strength"] = corr_pairs["Correlation"].apply(strength)
    corr_pairs["Correlation"] = corr_pairs["Correlation"].round(3)
    
    return corr_pairs[["Feature 1", "Feature 2", "Correlation", "Strength"]].head(top_n)


def get_distribution_info(series: pd.Series) -> dict:
    """
    Analyze the distribution of a numeric column.
    
    Returns info about:
    - Distribution type (normal, skewed, bimodal)
    - Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
    - Whether data looks normal
    """
    data = series.dropna()
    
    if len(data) < 8:
        return {"error": "Not enough data for distribution analysis"}
    
    # Choose normality test based on sample size
    # Shapiro-Wilk works best for n < 5000
    if len(data) < 5000:
        stat, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p_value = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"
    
    is_normal = p_value > 0.05
    
    skew = data.skew()
    if abs(skew) < 0.5:
        skew_desc = "Approximately symmetric"
    elif skew > 0:
        skew_desc = "Right-skewed (positive skew)"
    else:
        skew_desc = "Left-skewed (negative skew)"
    
    return {
        "test_name": test_name,
        "test_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "is_normal": is_normal,
        "skewness": round(skew, 3),
        "skew_description": skew_desc,
        "mean": round(data.mean(), 3),
        "median": round(data.median(), 3),
        "std": round(data.std(), 3),
    }