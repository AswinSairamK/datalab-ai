# ============================================================
# ml_explainer.py — Model explainability module
# ============================================================
# - SHAP values for feature impact
# - ROC curves for classification
# - Feature selection methods
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def get_shap_values(model, X_sample, max_samples: int = 100):
    """
    Compute SHAP values for model explainability.
    
    Returns SHAP values and feature importance.
    Uses TreeExplainer for tree models (fast) and KernelExplainer for others (slower).
    """
    try:
        import shap
        
        # Limit samples for performance
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=42)
        
        # Pick the right explainer based on model type
        model_type = type(model).__name__
        
        tree_models = [
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "DecisionTreeRegressor", "DecisionTreeClassifier",
            "XGBRegressor", "XGBClassifier"
        ]
        
        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        
        elif model_type in ["LinearRegression", "Ridge", "Lasso", "LogisticRegression"]:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        else:
            # Use KernelExplainer for any other model (slower)
            explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:50])
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        # Handle different SHAP output shapes:
        # - Regression: 2D array (samples × features)
        # - Binary classification: list of 2 arrays OR 3D array with shape (samples, features, 2)
        # - Multi-class: list of N arrays OR 3D array with shape (samples, features, N)
        shap_values = np.array(shap_values) if isinstance(shap_values, list) else shap_values
        
        if shap_values.ndim == 3:
            # Shape: (samples, features, classes) OR (classes, samples, features)
            # Normalize to (samples, features) by taking mean across classes for importance
            # but keep the class 1 (positive class) for binary classification
            if shap_values.shape[0] == len(X_sample):
                # Shape is (samples, features, classes) — use class 1 for binary, mean for multi
                if shap_values.shape[2] == 2:
                    shap_values = shap_values[:, :, 1]  # Positive class
                else:
                    shap_values = np.abs(shap_values).mean(axis=2)  # Average across classes
            else:
                # Shape is (classes, samples, features)
                if shap_values.shape[0] == 2:
                    shap_values = shap_values[1]  # Positive class for binary
                else:
                    shap_values = np.abs(shap_values).mean(axis=0)  # Average across classes
        
        # Ensure we have a 2D array
        shap_values = np.array(shap_values)
        if shap_values.ndim != 2:
            return {"error": f"Unexpected SHAP shape: {shap_values.shape}"}
        
        # Compute mean absolute SHAP for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "Feature": X_sample.columns,
            "SHAP Importance": mean_abs_shap
        }).sort_values("SHAP Importance", ascending=False)
        
        return {
            "shap_values": shap_values,
            "X_sample": X_sample,
            "importance_df": importance_df,
            "explainer_type": type(explainer).__name__
        }
    
    except Exception as e:
        return {"error": str(e)}


def get_roc_data(y_true, y_proba, classes: list) -> dict:
    """
    Compute ROC curve data for classification models.
    
    Returns fpr, tpr, and AUC for each class (or binary).
    """
    try:
        n_classes = len(classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc_score = auc(fpr, tpr)
            
            return {
                "binary": True,
                "fpr": fpr,
                "tpr": tpr,
                "auc": round(auc_score, 4),
                "classes": classes
            }
        
        else:
            # Multi-class: compute per-class ROC
            from sklearn.preprocessing import label_binarize
            y_binarized = label_binarize(y_true, classes=classes)
            
            roc_data = {}
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_binarized[:, i], y_proba[:, i])
                roc_data[str(class_label)] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": round(auc(fpr, tpr), 4)
                }
            
            return {
                "binary": False,
                "classes_data": roc_data,
                "classes": classes
            }
    
    except Exception as e:
        return {"error": str(e)}


def select_features_univariate(X, y, problem_type: str, k: int = 10) -> pd.DataFrame:
    """
    Select top k features using univariate statistical tests.
    """
    try:
        if problem_type == "regression":
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        
        selector.fit(X, y)
        
        scores_df = pd.DataFrame({
            "Feature": X.columns,
            "Score": selector.scores_,
            "P-value": selector.pvalues_,
            "Selected": selector.get_support()
        }).sort_values("Score", ascending=False)
        
        scores_df["Score"] = scores_df["Score"].round(3)
        scores_df["P-value"] = scores_df["P-value"].round(4)
        
        return scores_df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def select_features_mutual_info(X, y, problem_type: str, k: int = 10) -> pd.DataFrame:
    """
    Select features using mutual information.
    Captures non-linear relationships better than univariate tests.
    """
    try:
        if problem_type == "regression":
            scores = mutual_info_regression(X, y, random_state=42)
        else:
            scores = mutual_info_classif(X, y, random_state=42)
        
        scores_df = pd.DataFrame({
            "Feature": X.columns,
            "MI Score": scores
        }).sort_values("MI Score", ascending=False)
        
        scores_df["Selected"] = False
        scores_df.loc[scores_df.head(k).index, "Selected"] = True
        scores_df["MI Score"] = scores_df["MI Score"].round(4)
        
        return scores_df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def select_features_rfe(X, y, problem_type: str, k: int = 10) -> pd.DataFrame:
    """
    Recursive Feature Elimination using a tree model as estimator.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        if problem_type == "regression":
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]), step=1)
        selector.fit(X, y)
        
        scores_df = pd.DataFrame({
            "Feature": X.columns,
            "Rank": selector.ranking_,
            "Selected": selector.support_
        }).sort_values("Rank")
        
        return scores_df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def remove_low_variance_features(X, threshold: float = 0.01) -> pd.DataFrame:
    """
    Identify features with very low variance (almost constant).
    """
    try:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        scores_df = pd.DataFrame({
            "Feature": X.columns,
            "Variance": X.var().values,
            "Keep": selector.get_support()
        }).sort_values("Variance", ascending=False)
        
        scores_df["Variance"] = scores_df["Variance"].round(4)
        return scores_df
    
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def generate_profile_report_html(df: pd.DataFrame, title: str = "Data Profile") -> str:
    """
    Generate a simple data profile report using pandas + basic HTML.
    Lightweight alternative that works everywhere.
    """
    try:
        html = f"""
        <html><head><title>{title}</title>
        <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1 {{ color: #028090; }}
        h2 {{ color: #00A896; border-bottom: 2px solid #00A896; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th {{ background: #028090; color: white; padding: 8px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .stat {{ display: inline-block; padding: 10px 20px; margin: 5px;
                 background: #E1F5EE; border-radius: 8px; }}
        </style></head><body>
        <h1>{title}</h1>
        
        <h2>Dataset Overview</h2>
        <div class="stat"><b>Rows:</b> {len(df):,}</div>
        <div class="stat"><b>Columns:</b> {len(df.columns)}</div>
        <div class="stat"><b>Missing values:</b> {df.isnull().sum().sum():,}</div>
        <div class="stat"><b>Duplicates:</b> {df.duplicated().sum():,}</div>
        
        <h2>Column Details</h2>
        {df.describe(include='all').round(3).to_html()}
        
        <h2>Data Types</h2>
        {df.dtypes.to_frame('Type').to_html()}
        
        <h2>Missing Values</h2>
        {df.isnull().sum().to_frame('Missing Count').to_html()}
        
        <h2>First 20 Rows</h2>
        {df.head(20).to_html()}
        
        </body></html>
        """
        return html
    
    except Exception as e:
        return f"<p>Error generating report: {str(e)}</p>"