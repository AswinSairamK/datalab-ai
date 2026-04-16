# ============================================================
# data_loader.py — Handles file uploads and dataset loading
# ============================================================
# Supports CSV, Excel, and JSON files.
# Returns a pandas DataFrame for use throughout the app.
# ============================================================

import pandas as pd
import streamlit as st


def load_file(uploaded_file):
    """
    Load an uploaded file into a pandas DataFrame.
    
    Supports:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - JSON files (.json)
    
    Returns:
        tuple: (dataframe, error_message)
        If successful: (df, None)
        If failed: (None, error_message)
    """
    if uploaded_file is None:
        return None, "No file uploaded"
    
    file_name = uploaded_file.name.lower()
    
    try:
        if file_name.endswith('.csv'):
            # Try different encodings for CSV files
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        
        elif file_name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        else:
            return None, f"Unsupported file format: {file_name}"
        
        if df.empty:
            return None, "The uploaded file is empty"
        
        return df, None
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def load_sample_dataset(name: str):
    """
    Load a sample dataset for demo/testing purposes.
    
    Available datasets:
    - 'iris' — Iris flower dataset (classification)
    - 'titanic' — Titanic survival dataset (classification)
    - 'tips' — Restaurant tips dataset (regression)
    - 'diamonds' — Diamonds price dataset (regression)
    """
    try:
        if name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
            return df, None
        
        elif name == "titanic":
            # Use seaborn's built-in Titanic dataset
            import seaborn as sns
            df = sns.load_dataset('titanic')
            return df, None
        
        elif name == "tips":
            import seaborn as sns
            df = sns.load_dataset('tips')
            return df, None
        
        elif name == "diamonds":
            import seaborn as sns
            df = sns.load_dataset('diamonds')
            return df, None
        
        else:
            return None, f"Unknown sample dataset: {name}"
    
    except Exception as e:
        return None, f"Error loading sample: {str(e)}"


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about a dataset.
    
    Returns a dict with:
    - shape: (rows, columns)
    - memory_mb: memory usage in MB
    - num_numeric: count of numeric columns
    - num_categorical: count of categorical columns
    - num_datetime: count of datetime columns
    - total_missing: total number of missing values
    - missing_percentage: % of missing values
    - duplicate_rows: count of duplicate rows
    """
    num_rows, num_cols = df.shape
    
    # Memory usage in MB
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    
    # Column type counts
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    total_cells = num_rows * num_cols
    missing_percentage = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0
    
    # Duplicates
    duplicate_rows = df.duplicated().sum()
    
    return {
        "shape": (num_rows, num_cols),
        "rows": num_rows,
        "columns": num_cols,
        "memory_mb": memory_mb,
        "num_numeric": len(numeric_cols),
        "num_categorical": len(categorical_cols),
        "num_datetime": len(datetime_cols),
        "total_missing": int(total_missing),
        "missing_percentage": missing_percentage,
        "duplicate_rows": int(duplicate_rows),
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(categorical_cols),
        "datetime_columns": list(datetime_cols),
    }


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame showing info for each column.
    
    Returns a DataFrame with columns:
    - Column: column name
    - Type: data type
    - Non-Null: count of non-null values
    - Missing: count of missing values
    - Missing %: percentage of missing values
    - Unique: count of unique values
    - Sample: a sample value from the column
    """
    summary_data = []
    
    for col in df.columns:
        col_data = df[col]
        non_null_count = col_data.notna().sum()
        missing_count = col_data.isna().sum()
        missing_pct = round((missing_count / len(df) * 100), 2) if len(df) > 0 else 0
        unique_count = col_data.nunique()
        
        # Get a sample value (first non-null)
        sample = col_data.dropna().iloc[0] if non_null_count > 0 else "N/A"
        if isinstance(sample, (int, float)):
            sample = round(sample, 2)
        sample = str(sample)[:50]  # Truncate long values
        
        summary_data.append({
            "Column": col,
            "Type": str(col_data.dtype),
            "Non-Null": non_null_count,
            "Missing": missing_count,
            "Missing %": f"{missing_pct}%",
            "Unique": unique_count,
            "Sample": sample
        })
    
    return pd.DataFrame(summary_data)