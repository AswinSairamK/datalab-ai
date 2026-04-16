# ============================================================
# ml_evaluator.py — Model evaluation metrics
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def evaluate_regression(y_train, y_pred_train, y_test, y_pred_test) -> dict:
    """Compute regression metrics."""
    return {
        "R² (Train)": round(r2_score(y_train, y_pred_train), 4),
        "R² (Test)": round(r2_score(y_test, y_pred_test), 4),
        "RMSE (Train)": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
        "RMSE (Test)": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
        "MAE (Train)": round(mean_absolute_error(y_train, y_pred_train), 4),
        "MAE (Test)": round(mean_absolute_error(y_test, y_pred_test), 4),
        "MSE (Test)": round(mean_squared_error(y_test, y_pred_test), 4),
    }


def evaluate_classification(y_train, y_pred_train, y_test, y_pred_test, y_proba_test=None) -> dict:
    """Compute classification metrics."""
    avg = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    
    try:
        metrics = {
            "Accuracy (Train)": round(accuracy_score(y_train, y_pred_train), 4),
            "Accuracy (Test)": round(accuracy_score(y_test, y_pred_test), 4),
            "Precision (Test)": round(precision_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            "Recall (Test)": round(recall_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
            "F1 Score (Test)": round(f1_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
        }
        
        # ROC-AUC for binary classification
        if n_classes == 2 and y_proba_test is not None:
            try:
                auc = roc_auc_score(y_test, y_proba_test[:, 1])
                metrics["ROC-AUC (Test)"] = round(auc, 4)
            except:
                pass
        
        return metrics
    except Exception as e:
        return {"error": str(e)}


def get_confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    """Create a labeled confusion matrix DataFrame."""
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    df = pd.DataFrame(
        cm,
        index=[f"Actual {l}" for l in labels],
        columns=[f"Predicted {l}" for l in labels]
    )
    return df