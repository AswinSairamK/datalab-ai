# ============================================================
# ml_trainer.py — ML model training module
# ============================================================
# Handles training of regression and classification models.
# Auto-detects problem type based on target variable.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def detect_problem_type(y: pd.Series) -> str:
    """
    Auto-detect if this is a regression or classification problem.
    
    Returns: "regression" or "classification"
    """
    # If target is object/category or has few unique values → classification
    if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'bool':
        return "classification"
    
    # If numeric with few unique values → likely classification
    unique_count = y.nunique()
    if unique_count <= 10 and y.dtype in ['int64', 'int32']:
        return "classification"
    
    return "regression"


def get_available_models(problem_type: str) -> list:
    """Get list of available models for the problem type."""
    if problem_type == "regression":
        return [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "KNN",
            "SVR (Support Vector)"
        ]
    else:
        return [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "KNN",
            "SVM (Support Vector)"
        ]


def get_model(model_name: str, problem_type: str, params: dict = None):
    """Instantiate a model by name."""
    params = params or {}
    
    if problem_type == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        
        elif model_name == "Ridge Regression":
            return Ridge(alpha=params.get("alpha", 1.0), random_state=42)
        
        elif model_name == "Lasso Regression":
            return Lasso(alpha=params.get("alpha", 1.0), random_state=42)
        
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(
                max_depth=params.get("max_depth"),
                random_state=42
            )
        
        elif model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth"),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42
            )
        
        elif model_name == "KNN":
            return KNeighborsRegressor(
                n_neighbors=params.get("n_neighbors", 5),
                n_jobs=-1
            )
        
        elif model_name == "SVR (Support Vector)":
            return SVR(
                kernel=params.get("kernel", "rbf"),
                C=params.get("C", 1.0)
            )
    
    else:  # classification
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=params.get("max_depth"),
                random_state=42
            )
        
        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth"),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 3),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42
            )
        
        elif model_name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                n_jobs=-1
            )
        
        elif model_name == "SVM (Support Vector)":
            return SVC(
                kernel=params.get("kernel", "rbf"),
                C=params.get("C", 1.0),
                probability=True,
                random_state=42
            )
    
    return None


def prepare_data(df: pd.DataFrame, target: str, features: list, test_size: float = 0.2) -> dict:
    """
    Prepare data for training.
    
    - Drops rows with NaN in target or features
    - Splits into train/test
    - Returns X_train, X_test, y_train, y_test
    """
    # Select only target and features
    data = df[features + [target]].copy()
    
    # Drop rows with any NaN
    original_len = len(data)
    data = data.dropna()
    dropped = original_len - len(data)
    
    if len(data) < 10:
        return {"error": f"Not enough data after dropping NaN (only {len(data)} rows)"}
    
    # Check all features are numeric
    X = data[features]
    y = data[target]
    
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        return {"error": f"These features are not numeric: {non_numeric}. Please encode them first in Preprocessing."}
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if detect_problem_type(y) == "classification" and y.nunique() > 1 else None
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dropped_rows": dropped,
        "n_features": len(features),
        "problem_type": detect_problem_type(y)
    }


def train_model(model, X_train, y_train, X_test, y_test, problem_type: str) -> dict:
    """
    Train a model and return predictions + metrics.
    """
    import time
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 3)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    result = {
        "model": model,
        "train_time": train_time,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    # Get probabilities for classification if available
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            result["y_proba_test"] = model.predict_proba(X_test)
        except:
            pass
    
    return result


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Get feature importance from a trained model."""
    importances = None
    
    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Linear models
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = np.abs(coef).mean(axis=0)
        else:
            coef = np.abs(coef)
        importances = coef
    
    if importances is None:
        return pd.DataFrame()
    
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)