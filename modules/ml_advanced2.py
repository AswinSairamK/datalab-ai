# ============================================================
# ml_advanced2.py — Advanced ML: Clustering, Forecasting, PCA, SMOTE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CLUSTERING
# ============================================================

def run_kmeans(X, n_clusters: int = 3, random_state: int = 42) -> dict:
    """
    K-Means clustering.
    Returns cluster labels, centroids, and silhouette score.
    """
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette score (only if > 1 cluster)
        sil_score = None
        if n_clusters > 1 and len(np.unique(labels)) > 1:
            sil_score = round(silhouette_score(X_scaled, labels), 4)
        
        return {
            "method": "K-Means",
            "labels": labels,
            "centroids": kmeans.cluster_centers_,
            "inertia": round(kmeans.inertia_, 2),
            "silhouette": sil_score,
            "n_clusters": n_clusters,
            "scaled_X": X_scaled,
        }
    except Exception as e:
        return {"error": str(e)}


def run_dbscan(X, eps: float = 0.5, min_samples: int = 5) -> dict:
    """
    DBSCAN clustering (density-based).
    -1 = noise point.
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        
        sil_score = None
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
                sil_score = round(silhouette_score(X_scaled[mask], labels[mask]), 4)
        
        return {
            "method": "DBSCAN",
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": sil_score,
            "eps": eps,
            "min_samples": min_samples,
            "scaled_X": X_scaled,
        }
    except Exception as e:
        return {"error": str(e)}


def run_hierarchical(X, n_clusters: int = 3) -> dict:
    """Agglomerative (hierarchical) clustering."""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        hier = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hier.fit_predict(X_scaled)
        
        sil_score = None
        if n_clusters > 1:
            sil_score = round(silhouette_score(X_scaled, labels), 4)
        
        return {
            "method": "Hierarchical",
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette": sil_score,
            "scaled_X": X_scaled,
        }
    except Exception as e:
        return {"error": str(e)}


def find_optimal_k(X, max_k: int = 10) -> pd.DataFrame:
    """
    Find optimal number of clusters using elbow method and silhouette.
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
            
            results.append({
                "K": k,
                "Inertia": round(kmeans.inertia_, 2),
                "Silhouette": round(sil, 4)
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def run_pca(X, n_components: int = 2) -> dict:
    """
    PCA — Principal Component Analysis.
    Reduces dimensions while preserving variance.
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Component loadings (contribution of each original feature to each PC)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=X.columns
        )
        
        return {
            "method": "PCA",
            "X_reduced": X_pca,
            "explained_variance": explained_var.tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "loadings": loadings,
            "n_components": n_components,
        }
    except Exception as e:
        return {"error": str(e)}


def run_tsne(X, n_components: int = 2, perplexity: int = 30, max_samples: int = 2000) -> dict:
    """
    t-SNE — t-Distributed Stochastic Neighbor Embedding.
    Good for visualizing clusters in high-dimensional data.
    """
    try:
        # Sample for performance
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
            sampled_idx = X_sample.index
        else:
            X_sample = X
            sampled_idx = X.index
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Perplexity must be less than n_samples
        perplexity = min(perplexity, len(X_sample) - 1)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        return {
            "method": "t-SNE",
            "X_reduced": X_tsne,
            "n_components": n_components,
            "perplexity": perplexity,
            "sampled_idx": sampled_idx,
            "sampled_count": len(X_sample),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# IMBALANCED DATA (SMOTE)
# ============================================================

def apply_smote(X, y, sampling_strategy: str = "auto") -> dict:
    """
    SMOTE — Synthetic Minority Over-sampling Technique.
    Creates synthetic samples for minority classes.
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        # Count classes before
        before_counts = pd.Series(y).value_counts().to_dict()
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        after_counts = pd.Series(y_resampled).value_counts().to_dict()
        
        return {
            "X_resampled": X_resampled,
            "y_resampled": y_resampled,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "original_size": len(X),
            "new_size": len(X_resampled),
            "samples_added": len(X_resampled) - len(X),
        }
    except Exception as e:
        return {"error": str(e)}


def apply_undersampling(X, y) -> dict:
    """Random under-sampling of majority class."""
    try:
        from imblearn.under_sampling import RandomUnderSampler
        
        before_counts = pd.Series(y).value_counts().to_dict()
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        after_counts = pd.Series(y_resampled).value_counts().to_dict()
        
        return {
            "X_resampled": X_resampled,
            "y_resampled": y_resampled,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "original_size": len(X),
            "new_size": len(X_resampled),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# TIME SERIES FORECASTING
# ============================================================

def run_prophet_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30) -> dict:
    """
    Forecast using Facebook Prophet.
    
    df: DataFrame with date and value columns
    periods: number of future periods to forecast
    """
    try:
        from prophet import Prophet
        
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 10:
            return {"error": "Not enough data points (need at least 10)"}
        
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Split historical and future
        last_historical_date = prophet_df['ds'].max()
        
        historical = forecast[forecast['ds'] <= last_historical_date].copy()
        future_only = forecast[forecast['ds'] > last_historical_date].copy()
        
        return {
            "method": "Prophet",
            "historical_df": prophet_df,
            "forecast_df": forecast,
            "future_df": future_only,
            "historical_forecast": historical,
            "periods": periods,
            "components": ['trend', 'weekly', 'yearly']
        }
    except Exception as e:
        return {"error": str(e)}


def run_arima_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30) -> dict:
    """
    Forecast using ARIMA.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        ts_df = df[[date_col, value_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        ts_df = ts_df.dropna().sort_values(date_col)
        ts_df = ts_df.set_index(date_col)
        
        if len(ts_df) < 20:
            return {"error": "Not enough data points (need at least 20)"}
        
        # Fit ARIMA(1,1,1) as default
        model = ARIMA(ts_df[value_col], order=(1, 1, 1))
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=periods)
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=ts_df.index[-1], periods=periods + 1, freq='D')[1:],
            'forecast': forecast.values
        })
        
        return {
            "method": "ARIMA",
            "historical_df": ts_df.reset_index(),
            "forecast_df": forecast_df,
            "aic": round(fitted.aic, 2),
            "bic": round(fitted.bic, 2),
            "periods": periods,
            "date_col": date_col,
            "value_col": value_col
        }
    except Exception as e:
        return {"error": str(e)}