# ============================================================
# project_manager.py — Save/load projects + export notebooks
# ============================================================

import pandas as pd
import numpy as np
import pickle
import io
import json
from datetime import datetime


def save_project(df, df_processed, last_training, train_history) -> bytes:
    """
    Serialize the entire project state to bytes.
    Includes data, processed data, last trained model, and training history.
    """
    project = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "df": df,
        "df_processed": df_processed,
        "last_training": last_training,
        "train_history": train_history,
    }
    
    buffer = io.BytesIO()
    pickle.dump(project, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_project(file_bytes) -> dict:
    """Load a saved project from bytes."""
    try:
        buffer = io.BytesIO(file_bytes)
        project = pickle.load(buffer)
        return {"success": True, "project": project}
    except Exception as e:
        return {"success": False, "error": str(e)}


def export_as_notebook(
    dataset_name: str,
    preprocessing_steps: list,
    feature_engineering_steps: list,
    target: str,
    features: list,
    model_name: str,
    problem_type: str
) -> str:
    """
    Generate a Jupyter notebook (.ipynb JSON) that reproduces the workflow.
    """
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# DataLab AI Export — {dataset_name}\n",
            f"\n",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"\n",
            f"**Target:** `{target}`  \n",
            f"**Model:** `{model_name}`  \n",
            f"**Problem type:** {problem_type}\n"
        ]
    })
    
    # Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split, cross_val_score\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report\n",
            "\n",
            "%matplotlib inline"
        ]
    })
    
    # Load data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Load data"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# Load your dataset\n",
            f"df = pd.read_csv('{dataset_name}.csv')  # Update path as needed\n",
            f"print(f'Shape: {{df.shape}}')\n",
            f"df.head()"
        ]
    })
    
    # EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Exploratory data analysis"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Basic statistics\n",
            "df.describe()\n"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation heatmap\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')\n",
            "plt.title('Correlation Matrix')\n",
            "plt.show()"
        ]
    })
    
    # Preprocessing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Preprocessing"]
    })
    pp_code = ["# Preprocessing steps applied:\n"]
    for step in preprocessing_steps:
        pp_code.append(f"# - {step}\n")
    pp_code.append("\n")
    pp_code.append("# Fill missing values\n")
    pp_code.append("df = df.fillna(df.mean(numeric_only=True))\n")
    pp_code.append("\n")
    pp_code.append("# Encode categorical columns\n")
    pp_code.append("for col in df.select_dtypes(include='object').columns:\n")
    pp_code.append("    df[col] = LabelEncoder().fit_transform(df[col].astype(str))")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": pp_code
    })
    
    # Train/test split
    features_str = ", ".join([f"'{f}'" for f in features])
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Train/test split"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"features = [{features_str}]\n",
            f"target = '{target}'\n",
            f"\n",
            f"X = df[features]\n",
            f"y = df[target]\n",
            f"\n",
            f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            f"print(f'Train: {{X_train.shape}} | Test: {{X_test.shape}}')"
        ]
    })
    
    # Model training
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## 5. Train {model_name}"]
    })
    
    model_import_map = {
        "Linear Regression": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()",
        "Ridge Regression": "from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)",
        "Lasso Regression": "from sklearn.linear_model import Lasso\nmodel = Lasso(alpha=1.0)",
        "Decision Tree": f"from sklearn.tree import Decision{'Tree' if problem_type == 'regression' else 'Tree'}{'Regressor' if problem_type == 'regression' else 'Classifier'}\nmodel = Decision{'Tree' if problem_type == 'regression' else 'Tree'}{'Regressor' if problem_type == 'regression' else 'Classifier'}()",
        "Random Forest": f"from sklearn.ensemble import RandomForest{'Regressor' if problem_type == 'regression' else 'Classifier'}\nmodel = RandomForest{'Regressor' if problem_type == 'regression' else 'Classifier'}(n_estimators=100, random_state=42)",
        "Gradient Boosting": f"from sklearn.ensemble import GradientBoosting{'Regressor' if problem_type == 'regression' else 'Classifier'}\nmodel = GradientBoosting{'Regressor' if problem_type == 'regression' else 'Classifier'}()",
        "KNN": f"from sklearn.neighbors import KNeighbors{'Regressor' if problem_type == 'regression' else 'Classifier'}\nmodel = KNeighbors{'Regressor' if problem_type == 'regression' else 'Classifier'}(n_neighbors=5)",
        "Logistic Regression": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)",
    }
    
    model_code = model_import_map.get(model_name, "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()")
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            model_code + "\n",
            "\n",
            "model.fit(X_train, y_train)\n",
            "y_pred = model.predict(X_test)"
        ]
    })
    
    # Evaluation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Evaluate"]
    })
    
    if problem_type == "regression":
        eval_code = [
            "r2 = r2_score(y_test, y_pred)\n",
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
            "print(f'R² Score: {r2:.4f}')\n",
            "print(f'RMSE: {rmse:.4f}')\n",
            "\n",
            "# Actual vs Predicted plot\n",
            "plt.figure(figsize=(8, 6))\n",
            "plt.scatter(y_test, y_pred, alpha=0.6)\n",
            "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')\n",
            "plt.xlabel('Actual')\n",
            "plt.ylabel('Predicted')\n",
            "plt.title('Actual vs Predicted')\n",
            "plt.show()"
        ]
    else:
        eval_code = [
            "acc = accuracy_score(y_test, y_pred)\n",
            "print(f'Accuracy: {acc:.4f}')\n",
            "print(classification_report(y_test, y_pred))"
        ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": eval_code
    })
    
    # Feature importance (if tree model)
    if "Forest" in model_name or "Boosting" in model_name or "Tree" in model_name:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Feature importance"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importance_df = pd.DataFrame({\n",
                "    'Feature': features,\n",
                "    'Importance': model.feature_importances_\n",
                "}).sort_values('Importance', ascending=False)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=importance_df, x='Importance', y='Feature')\n",
                "plt.title('Feature Importance')\n",
                "plt.show()\n",
                "\n",
                "importance_df"
            ]
        })
    
    # Build notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return json.dumps(notebook, indent=2)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_numeric_data(df: pd.DataFrame, columns: list, method: str = "noise", n_samples: int = 100, noise_level: float = 0.05) -> dict:
    """
    Generate synthetic data samples.
    
    Methods:
    - "noise": add Gaussian noise to existing samples
    - "interpolation": linear interpolation between random pairs
    - "bootstrap": bootstrap resampling
    """
    try:
        if not columns:
            return {"error": "No columns selected"}
        
        # Get valid data
        valid_df = df[columns].dropna()
        
        if len(valid_df) < 2:
            return {"error": "Not enough data"}
        
        if method == "noise":
            # Randomly sample rows and add noise
            sampled = valid_df.sample(n=n_samples, replace=True, random_state=42).copy()
            stds = valid_df.std()
            
            for col in columns:
                noise = np.random.normal(0, stds[col] * noise_level, size=n_samples)
                sampled[col] = sampled[col] + noise
            
            synthetic = sampled.reset_index(drop=True)
        
        elif method == "interpolation":
            # Pick random pairs and interpolate
            synthetic_rows = []
            for _ in range(n_samples):
                idx1, idx2 = np.random.choice(valid_df.index, 2, replace=False)
                alpha = np.random.random()
                new_row = alpha * valid_df.loc[idx1] + (1 - alpha) * valid_df.loc[idx2]
                synthetic_rows.append(new_row)
            
            synthetic = pd.DataFrame(synthetic_rows).reset_index(drop=True)
        
        elif method == "bootstrap":
            # Simple bootstrap sampling
            synthetic = valid_df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
        
        else:
            return {"error": f"Unknown method: {method}"}
        
        # Mark synthetic rows
        synthetic["_synthetic"] = True
        
        # Create combined dataset
        original_marked = df.copy()
        original_marked["_synthetic"] = False
        combined = pd.concat([original_marked, synthetic], ignore_index=True)
        
        return {
            "synthetic_df": synthetic,
            "combined_df": combined,
            "original_size": len(df),
            "new_size": len(combined),
            "n_synthetic": len(synthetic),
            "method": method
        }
    except Exception as e:
        return {"error": str(e)}