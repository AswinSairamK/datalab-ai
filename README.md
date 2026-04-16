# DataLab AI

> Complete Data Science Workbench — 21 features across 11 workflow stages, from upload to trained model

DataLab AI is a no-code data science platform that lets anyone upload a dataset and go from raw CSV to trained ML model in minutes. Built entirely with Python and Streamlit, it combines EDA, preprocessing, Auto-ML, SHAP explanations, clustering, forecasting, and more in a single unified interface.

**🚀 Live Demo:** https://aswin-datalab.streamlit.app

## Features

### Data Handling
- **Upload** CSV, Excel, JSON or use sample datasets (iris, titanic, tips, diamonds)
- **Overview** with shape, dtypes, missing values, duplicates, and column details

### Exploratory Data Analysis (5 sub-tabs)
- Descriptive statistics (mean, median, IQR, skewness, kurtosis)
- Distributions with normality tests (Shapiro-Wilk / D'Agostino)
- Correlation heatmaps (Pearson, Spearman, Kendall)
- Categorical analysis with bar charts
- Advanced plots: scatter, scatter matrix, 3D scatter, violin, parallel coordinates, density

### Outlier Detection
- IQR, Z-score, Modified Z-score (univariate)
- Isolation Forest, LOF (multivariate ML-based)
- Method comparison view

### Preprocessing
- Missing value handling (mean, median, mode, custom, drop)
- Categorical encoding (label, one-hot, ordinal)
- Feature scaling (standard, min-max, robust)
- Drop columns, remove duplicates, convert types

### Feature Engineering
- Math transforms (log, sqrt, square, reciprocal, etc.)
- Binning (equal width, equal frequency)
- Interactions (multiply, add, subtract, divide, ratio)
- Polynomial features
- Datetime extraction
- Feature selection (F-test, mutual info, RFE)

### ML Training
- **Manual training** with 14+ models
- **Auto-ML** — trains all models, ranks by score
- **Cross-validation** with multiple metrics
- **Hyperparameter tuning** via GridSearchCV
- **Train history** to compare session models

### Model Explainability
- SHAP values (TreeExplainer, LinearExplainer, KernelExplainer)
- ROC curves for classification (binary + multi-class)
- Feature importance visualization

### Advanced Analytics
- **Clustering:** K-Means, DBSCAN, Hierarchical
- **Dimensionality reduction:** PCA, t-SNE
- **Forecasting:** Prophet, ARIMA
- **Imbalanced data:** SMOTE oversampling, random undersampling

### Predictions & Export
- Manual input (single prediction with confidence scores)
- Batch predictions from uploaded file
- Predict on current dataset with R²/accuracy comparison
- Export trained models as .pkl
- Export full workflow as Jupyter notebook
- Save/load projects as .datalab files
- Data augmentation (Gaussian noise, interpolation, bootstrap)

## Tech Stack

- **Frontend:** Streamlit + Plotly
- **Data:** pandas, numpy, scipy, statsmodels
- **ML:** scikit-learn (14+ models)
- **Explainability:** SHAP
- **Forecasting:** Prophet, ARIMA (statsmodels)
- **Imbalanced data:** imbalanced-learn
- **Visualization:** Plotly, seaborn, matplotlib

## Quick Start

\\\ash
# Clone the repo
git clone https://github.com/AswinSairamK/datalab-ai.git
cd datalab-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
\\\

Open your browser at http://localhost:8501

## Usage

1. **Upload a dataset** (CSV, Excel, JSON) or pick a sample
2. **Explore** with the EDA tab — stats, distributions, correlations
3. **Find outliers** using statistical or ML-based methods
4. **Preprocess** your data (missing values, encoding, scaling)
5. **Engineer features** with transforms and interactions
6. **Train models** manually or let Auto-ML pick the best
7. **Understand** your model with SHAP and ROC curves
8. **Explore further** with clustering, PCA, or forecasting
9. **Predict** on new data or export the model
10. **Save** your entire project or export as a Jupyter notebook

## Project Structure

\\\
datalab-ai/
├── app.py                       # Main Streamlit entry
├── modules/
│   ├── data_loader.py          # File upload, sample loading
│   ├── eda.py                  # Statistics, distributions, correlations
│   ├── outliers.py             # 5 outlier detection methods
│   ├── preprocessing.py        # Missing values, encoding, scaling
│   ├── feature_engineering.py  # Transforms, interactions, datetime
│   ├── ml_trainer.py           # Model training and preparation
│   ├── ml_evaluator.py         # Metrics (R², accuracy, confusion matrix)
│   ├── ml_advanced.py          # Auto-ML, CV, hyperparameter tuning
│   ├── ml_advanced2.py         # Clustering, PCA, t-SNE, forecasting, SMOTE
│   ├── ml_explainer.py         # SHAP, ROC, feature selection
│   └── project_manager.py      # Save/load, notebook export, augmentation
└── data/sample_datasets/       # Sample data files
\\\

## Screenshots

_Add screenshots of your app here_

## Author

**Aswin Sairam Kannan**  
Data Engineer | Chennai, India  
Email: sairam111297@gmail.com  
GitHub: [@AswinSairamK](https://github.com/AswinSairamK)

## License

MIT License — free to use, modify, and share.
