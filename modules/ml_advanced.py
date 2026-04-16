# ============================================================
# ml_advanced.py — Advanced ML features
# ============================================================
# - Auto-ML: train all models at once
# - Cross-validation
# - Hyperparameter tuning (GridSearchCV)
# - Train history tracking
# ============================================================

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from modules.ml_trainer import get_model, get_available_models


def run_auto_ml(X_train, y_train, X_test, y_test, problem_type: str, cv_folds: int = 5) -> pd.DataFrame:
    """
    Train ALL available models and rank them by performance.
    
    Returns a DataFrame with results for each model, sorted best-to-worst.
    """
    models = get_available_models(problem_type)
    results = []
    
    for model_name in models:
        try:
            start = time.time()
            model = get_model(model_name, problem_type)
            
            if model is None:
                continue
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Cross-validation score
            if problem_type == "regression":
                cv_scoring = "r2"
            else:
                cv_scoring = "accuracy"
            
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=cv_scoring, n_jobs=-1)
                cv_mean = round(cv_scores.mean(), 4)
                cv_std = round(cv_scores.std(), 4)
            except Exception:
                cv_mean = None
                cv_std = None
            
            train_time = round(time.time() - start, 3)
            
            if problem_type == "regression":
                result = {
                    "Model": model_name,
                    "R² (Test)": round(r2_score(y_test, y_pred_test), 4),
                    "R² (Train)": round(r2_score(y_train, y_pred_train), 4),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
                    "MAE": round(mean_absolute_error(y_test, y_pred_test), 4),
                    "CV Score": cv_mean,
                    "CV Std": cv_std,
                    "Time (s)": train_time,
                    "_model": model,  # Store the actual model
                }
            else:
                avg = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
                result = {
                    "Model": model_name,
                    "Accuracy (Test)": round(accuracy_score(y_test, y_pred_test), 4),
                    "Accuracy (Train)": round(accuracy_score(y_train, y_pred_train), 4),
                    "Precision": round(precision_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
                    "F1 Score": round(f1_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
                    "CV Score": cv_mean,
                    "CV Std": cv_std,
                    "Time (s)": train_time,
                    "_model": model,
                }
            
            results.append(result)
        
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
            continue
    
    # Sort by the main metric
    df = pd.DataFrame(results)
    if not df.empty:
        sort_col = "R² (Test)" if problem_type == "regression" else "Accuracy (Test)"
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    
    return df


def run_cross_validation(model, X, y, cv_folds: int, problem_type: str) -> dict:
    """
    Run k-fold cross-validation on a model.
    
    Returns mean, std, and all scores across folds.
    """
    if problem_type == "regression":
        scoring_options = {
            "R²": "r2",
            "Neg MSE": "neg_mean_squared_error",
            "Neg MAE": "neg_mean_absolute_error"
        }
    else:
        scoring_options = {
            "Accuracy": "accuracy",
            "Precision": "precision_weighted",
            "Recall": "recall_weighted",
            "F1": "f1_weighted"
        }
    
    results = {}
    
    for metric_name, scoring in scoring_options.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            
            # Negate the neg_ metrics for display
            if "Neg" in metric_name:
                metric_name = metric_name.replace("Neg ", "")
                scores = -scores
                if "MSE" in metric_name:
                    scores = np.sqrt(scores)
                    metric_name = "RMSE"
            
            results[metric_name] = {
                "mean": round(scores.mean(), 4),
                "std": round(scores.std(), 4),
                "scores": [round(s, 4) for s in scores],
                "min": round(scores.min(), 4),
                "max": round(scores.max(), 4)
            }
        except Exception as e:
            continue
    
    return results


def get_hyperparameter_grid(model_name: str, problem_type: str) -> dict:
    """
    Get a reasonable hyperparameter grid for GridSearchCV.
    Kept small to complete in reasonable time.
    """
    grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "Decision Tree": {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 10, 15],
            "weights": ["uniform", "distance"]
        },
        "Ridge Regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "Lasso Regression": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        "SVR (Support Vector)": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"]
        },
        "SVM (Support Vector)": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0]
        }
    }
    
    return grids.get(model_name, {})


def run_hyperparameter_tuning(model, X_train, y_train, param_grid: dict, cv_folds: int, problem_type: str) -> dict:
    """
    Run GridSearchCV to find the best hyperparameters.
    """
    if not param_grid:
        return {"error": "No parameter grid available for this model"}
    
    scoring = "r2" if problem_type == "regression" else "accuracy"
    
    try:
        grid = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        start = time.time()
        grid.fit(X_train, y_train)
        search_time = round(time.time() - start, 2)
        
        # Get top 5 combinations
        results_df = pd.DataFrame(grid.cv_results_)
        results_df = results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        results_df = results_df.sort_values("rank_test_score").head(5)
        results_df["mean_test_score"] = results_df["mean_test_score"].round(4)
        results_df["std_test_score"] = results_df["std_test_score"].round(4)
        
        return {
            "best_params": grid.best_params_,
            "best_score": round(grid.best_score_, 4),
            "best_model": grid.best_estimator_,
            "total_combinations": len(grid.cv_results_["params"]),
            "search_time": search_time,
            "top_5": results_df,
        }
    except Exception as e:
        return {"error": str(e)}