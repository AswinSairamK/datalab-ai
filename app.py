# ============================================================
# DataLab AI — Main Streamlit Application
# ============================================================
# Entry point for the data science workbench.
# Handles navigation and session state.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_file, load_sample_dataset, get_dataset_info, get_column_summary


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="DataLab AI — Data Science Workbench",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Session State Initialization
# ============================================================
# Session state persists data across reruns.
# We store the uploaded DataFrame here so all tabs can access it.
if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None


# ============================================================
# Custom CSS for a cleaner look
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E2761;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00A896;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================
st.markdown('<p class="main-header">🧪 DataLab AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your personal data science workbench — upload any dataset and analyze it from EDA to ML</p>', unsafe_allow_html=True)


# ============================================================
# Sidebar — Data Source Selection
# ============================================================
with st.sidebar:
    st.header("📁 Data Source")
    
    # Radio button to choose between upload and sample
    data_source = st.radio(
        "Choose data source:",
        ["Upload file", "Use sample dataset"],
        help="Upload your own file or try a built-in dataset"
    )
    
    if data_source == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, or JSON",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading dataset..."):
                df, error = load_file(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.session_state.df_name = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name}")
    
    else:  # Sample dataset
        sample_name = st.selectbox(
            "Choose a sample dataset:",
            ["iris", "titanic", "tips", "diamonds"],
            help="Pre-loaded datasets for testing"
        )
        
        if st.button("Load sample", use_container_width=True):
            with st.spinner(f"Loading {sample_name}..."):
                df, error = load_sample_dataset(sample_name)
            
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.session_state.df_name = f"{sample_name} (sample)"
                st.success(f"Loaded: {sample_name}")
    
    # Show current dataset info
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 📊 Current Dataset")
        st.info(f"**{st.session_state.df_name}**\n\n"
                f"Rows: {st.session_state.df.shape[0]:,}\n\n"
                f"Columns: {st.session_state.df.shape[1]}")
        
        if st.button("🗑️ Clear dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_name = None
            st.rerun()


# ============================================================
# Main Content Area
# ============================================================
if st.session_state.df is None:
    # Welcome screen when no data is loaded
    st.info("👈 **Get started:** Upload a file or load a sample dataset from the sidebar")
    
    st.markdown("### What you can do with DataLab AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Explore & Analyze**
        - Dataset overview and statistics
        - Distribution plots
        - Correlation heatmaps
        - Outlier detection
        """)
    
    with col2:
        st.markdown("""
        **🛠️ Preprocess & Engineer**
        - Handle missing values
        - Encode categories
        - Scale features
        - Create new features
        """)
    
    with col3:
        st.markdown("""
        **🤖 Build & Deploy ML**
        - Train models from scratch
        - Pick target and features
        - Evaluate with R², accuracy
        - Export trained models
        """)
    
    st.markdown("---")
    st.markdown("### 📚 Try a sample dataset")
    
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        st.markdown("**🌸 Iris**\n\nClassification\n\n150 rows")
    with sample_col2:
        st.markdown("**🚢 Titanic**\n\nClassification\n\n891 rows")
    with sample_col3:
        st.markdown("**💰 Tips**\n\nRegression\n\n244 rows")
    with sample_col4:
        st.markdown("**💎 Diamonds**\n\nRegression\n\n53,940 rows")

else:
    # Dataset is loaded — show the analysis tabs
    df = st.session_state.df
    
    # Create tabs for different analysis sections
    (tab_overview, tab_eda, tab_outliers, tab_preprocessing, tab_fe,
     tab_ml, tab_predict, tab_cluster, tab_dimred, tab_forecast, tab_project) = st.tabs([
        "📊 Overview",
        "🔍 EDA",
        "⚠️ Outliers",
        "🛠️ Preprocessing",
        "⚡ Feature Engineering",
        "🤖 ML Training",
        "🔮 Predictions",
        "🎯 Clustering",
        "📐 Dim. Reduction",
        "📅 Forecasting",
        "💼 Project"
    ])
    
    # ============================================================
    # TAB 1: Overview
    # ============================================================
    with tab_overview:
        st.subheader("Dataset Overview")
        
        # Get dataset info
        info = get_dataset_info(df)
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{info['rows']:,}")
        col2.metric("Columns", info['columns'])
        col3.metric("Memory", f"{info['memory_mb']} MB")
        col4.metric("Duplicates", info['duplicate_rows'])
        
        st.markdown("---")
        
        # Column type breakdown
        col1, col2, col3 = st.columns(3)
        col1.metric("🔢 Numeric columns", info['num_numeric'])
        col2.metric("📝 Categorical columns", info['num_categorical'])
        col3.metric("📅 Datetime columns", info['num_datetime'])
        
        st.markdown("---")
        
        # Missing values summary
        st.subheader("Missing Values")
        col1, col2 = st.columns(2)
        col1.metric("Total missing", f"{info['total_missing']:,}")
        col2.metric("Missing percentage", f"{info['missing_percentage']}%")
        
        st.markdown("---")
        
        # Column summary table
        st.subheader("Column Details")
        column_summary = get_column_summary(df)
        st.dataframe(
            column_summary,
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Data preview
        st.subheader("Data Preview")
        
        preview_col1, preview_col2 = st.columns([3, 1])
        with preview_col2:
            n_rows = st.slider("Rows to show", min_value=5, max_value=100, value=10, step=5)
        
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        # ----- Data Profiling Report -----
        st.markdown("---")
        st.subheader("📋 Full Data Profile Report")
        st.caption("Generate a comprehensive HTML report with all statistics, distributions, and correlations")
        
        if st.button("📊 Generate full profile report", type="primary", key="profile_btn"):
            with st.spinner("Generating comprehensive profile report... (this may take a minute)"):
                from modules.ml_explainer import generate_profile_report_html
                
                # Sample large datasets for performance
                if len(df) > 10000:
                    st.caption("Sampling 10,000 rows for performance")
                    profile_df = df.sample(n=10000, random_state=42)
                else:
                    profile_df = df
                
                html_report = generate_profile_report_html(profile_df, title="DataLab AI Profile")
                st.session_state.profile_html = html_report
                st.success("✅ Report generated!")
        
        if "profile_html" in st.session_state:
            import streamlit.components.v1 as components
            components.html(st.session_state.profile_html, height=800, scrolling=True)
            
            st.download_button(
                label="📥 Download report as HTML",
                data=st.session_state.profile_html,
                file_name="datalab_profile_report.html",
                mime="text/html"
            )
        
        # Data shape summary
        st.caption(f"Showing first {n_rows} of {info['rows']:,} rows")
    
   # ============================================================
    # TAB 2: EDA (Exploratory Data Analysis)
    # ============================================================
    with tab_eda:
        from modules.eda import (
            get_descriptive_stats, get_categorical_stats,
            get_correlation_matrix, get_top_correlations,
            get_distribution_info
        )
        import plotly.express as px
        import plotly.graph_objects as go
        
        st.subheader("Exploratory Data Analysis")
        
        # Sub-tabs for different EDA views
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "📊 Descriptive Stats",
            "📈 Distributions",
            "🔗 Correlations",
            "📝 Categorical Analysis",
            "🎨 Advanced Plots"
        ])
        
        # ----- EDA Sub-tab 1: Descriptive Statistics -----
        with eda_tab1:
            st.markdown("### Descriptive statistics for numeric columns")
            
            desc_stats = get_descriptive_stats(df)
            
            if desc_stats.empty:
                st.warning("No numeric columns found in the dataset")
            else:
                st.dataframe(desc_stats, use_container_width=True, hide_index=True)
                
                # Explanation
                with st.expander("📖 What do these statistics mean?"):
                    st.markdown("""
                    - **Mean**: Average value (affected by outliers)
                    - **Median**: Middle value (robust to outliers)
                    - **Std**: Standard deviation — how spread out the values are
                    - **Variance**: Square of standard deviation
                    - **IQR**: Interquartile range (Q3 - Q1) — middle 50% spread
                    - **Skewness**: Measure of asymmetry
                        - 0 = symmetric (normal)
                        - positive = right-skewed (tail on right)
                        - negative = left-skewed (tail on left)
                    - **Kurtosis**: "Tailedness" of the distribution
                        - 0 = normal
                        - positive = heavy tails (more outliers)
                        - negative = light tails (fewer outliers)
                    """)
        
        # ----- EDA Sub-tab 2: Distributions -----
        with eda_tab2:
            st.markdown("### Distribution analysis")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns to analyze")
            else:
                selected_col = st.selectbox(
                    "Select a numeric column:",
                    numeric_cols,
                    key="dist_col"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(
                        df, x=selected_col,
                        nbins=30,
                        title=f"Distribution of {selected_col}",
                        color_discrete_sequence=['#00A896']
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        df, y=selected_col,
                        title=f"Box plot of {selected_col}",
                        color_discrete_sequence=['#1E2761']
                    )
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Distribution statistics
                st.markdown("#### Distribution analysis")
                dist_info = get_distribution_info(df[selected_col])
                
                if "error" in dist_info:
                    st.warning(dist_info["error"])
                else:
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    stat_col1.metric("Mean", dist_info["mean"])
                    stat_col2.metric("Median", dist_info["median"])
                    stat_col3.metric("Std Dev", dist_info["std"])
                    stat_col4.metric("Skewness", dist_info["skewness"])
                    
                    # Normality test
                    st.markdown("#### Normality test")
                    test_col1, test_col2 = st.columns(2)
                    test_col1.info(
                        f"**Test:** {dist_info['test_name']}\n\n"
                        f"**Statistic:** {dist_info['test_statistic']}\n\n"
                        f"**P-value:** {dist_info['p_value']}"
                    )
                    
                    if dist_info["is_normal"]:
                        test_col2.success(
                            f"✅ **Data appears normally distributed**\n\n"
                            f"{dist_info['skew_description']}"
                        )
                    else:
                        test_col2.warning(
                            f"⚠️ **Data is NOT normally distributed**\n\n"
                            f"{dist_info['skew_description']}"
                        )
        
        # ----- EDA Sub-tab 3: Correlations -----
        with eda_tab3:
            st.markdown("### Correlation analysis")
            
            corr_method = st.selectbox(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                help="Pearson: linear, Spearman: rank-based, Kendall: tau"
            )
            
            corr_matrix = get_correlation_matrix(df, method=corr_method)
            
            if corr_matrix.empty:
                st.warning("Need at least 2 numeric columns for correlation")
            else:
                # Correlation heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title=f"{corr_method.capitalize()} Correlation Heatmap"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Top correlations
                st.markdown("#### Strongest correlations")
                top_corr = get_top_correlations(corr_matrix, top_n=10)
                
                if not top_corr.empty:
                    st.dataframe(top_corr, use_container_width=True, hide_index=True)
                
                with st.expander("📖 How to interpret correlations"):
                    st.markdown("""
                    **Correlation ranges from -1 to +1:**
                    - **+1**: Perfect positive correlation (both increase together)
                    - **0**: No linear relationship
                    - **-1**: Perfect negative correlation (one increases as other decreases)
                    
                    **Strength guide:**
                    - **0.7 to 1.0**: Strong correlation
                    - **0.4 to 0.7**: Moderate correlation
                    - **0.2 to 0.4**: Weak correlation
                    - **Below 0.2**: Very weak or no correlation
                    
                    **Note:** Correlation doesn't imply causation!
                    """)
        
        # ----- EDA Sub-tab 4: Categorical Analysis -----
        with eda_tab4:
            st.markdown("### Categorical column analysis")
            
            cat_stats = get_categorical_stats(df)
            
            if not cat_stats:
                st.warning("No categorical columns found in the dataset")
            else:
                selected_cat = st.selectbox(
                    "Select a categorical column:",
                    list(cat_stats.keys()),
                    key="cat_col"
                )
                
                cat_data = cat_stats[selected_cat]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart of top values
                    fig_bar = px.bar(
                        cat_data.head(10),
                        x="Value",
                        y="Count",
                        title=f"Top values in {selected_cat}",
                        color_discrete_sequence=['#00A896']
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    st.markdown("#### Value counts")
                    st.dataframe(cat_data, use_container_width=True, hide_index=True)
                
                # Stats
                unique_count = df[selected_cat].nunique()
                missing_count = df[selected_cat].isna().sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Unique values", unique_count)
                col2.metric("Missing values", int(missing_count))
                col3.metric("Most common", cat_data.iloc[0]["Value"] if not cat_data.empty else "N/A")
                
            # ----- EDA Sub-tab 5: Advanced Plots -----
        with eda_tab5:
            st.markdown("### Advanced visualizations")
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            if not numeric_cols:
                st.warning("Need numeric columns for these plots")
            else:
                plot_type = st.selectbox(
                    "Choose plot type:",
                    [
                        "Scatter Plot",
                        "Scatter Matrix (Pair Plot)",
                        "3D Scatter Plot",
                        "Violin Plot",
                        "Categorical vs Numeric",
                        "Density Plot",
                        "Parallel Coordinates"
                    ],
                    key="advanced_plot_type"
                )
                
                # ----- Scatter Plot -----
                if plot_type == "Scatter Plot":
                    st.markdown("Visualize the relationship between two numeric variables")
                    
                    if len(numeric_cols) < 2:
                        st.warning("Need at least 2 numeric columns")
                    else:
                        col1, col2, col3 = st.columns(3)
                        x_axis = col1.selectbox("X axis:", numeric_cols, key="scatter_x")
                        y_axis = col2.selectbox("Y axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
                        
                        color_options = ["None"] + cat_cols + numeric_cols
                        color_by = col3.selectbox("Color by:", color_options, key="scatter_color")
                        
                        # Build scatter plot
                        scatter_kwargs = {
                            "x": x_axis,
                            "y": y_axis,
                            "title": f"{y_axis} vs {x_axis}",
                            "opacity": 0.6,
                            "trendline": "ols" if st.checkbox("Show trend line", key="scatter_trend") else None
                        }
                        
                        if color_by != "None":
                            scatter_kwargs["color"] = color_by
                        else:
                            scatter_kwargs["color_discrete_sequence"] = ['#00A896']
                        
                        try:
                            fig = px.scatter(df, **scatter_kwargs)
                        except:
                            scatter_kwargs.pop("trendline", None)
                            fig = px.scatter(df, **scatter_kwargs)
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show correlation between the two variables
                        corr_value = df[x_axis].corr(df[y_axis])
                        st.info(f"**Correlation between {x_axis} and {y_axis}:** {round(corr_value, 3)}")
                
                # ----- Scatter Matrix (Pair Plot) -----
                elif plot_type == "Scatter Matrix (Pair Plot)":
                    st.markdown("See relationships between multiple numeric variables at once")
                    
                    selected_cols = st.multiselect(
                        "Select columns (3-6 recommended):",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key="pair_cols"
                    )
                    
                    color_options = ["None"] + cat_cols
                    color_by = st.selectbox("Color by category:", color_options, key="pair_color")
                    
                    if len(selected_cols) >= 2:
                        if len(selected_cols) > 6:
                            st.warning("Too many columns selected — showing first 6 for performance")
                            selected_cols = selected_cols[:6]
                        
                        sample_size = min(1000, len(df))
                        plot_df = df[selected_cols].sample(n=sample_size) if len(df) > sample_size else df[selected_cols]
                        
                        if color_by != "None":
                            plot_df = pd.concat([plot_df, df[color_by]], axis=1)
                            fig = px.scatter_matrix(
                                plot_df,
                                dimensions=selected_cols,
                                color=color_by,
                                title="Scatter Matrix"
                            )
                        else:
                            fig = px.scatter_matrix(
                                plot_df,
                                dimensions=selected_cols,
                                title="Scatter Matrix",
                                color_discrete_sequence=['#00A896']
                            )
                        
                        fig.update_layout(height=600)
                        fig.update_traces(diagonal_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(df) > 1000:
                            st.caption(f"Showing 1,000 sampled rows for performance (total: {len(df):,})")
                    else:
                        st.info("Select at least 2 columns")
                
                # ----- 3D Scatter Plot -----
                elif plot_type == "3D Scatter Plot":
                    st.markdown("Visualize 3 variables in 3D space")
                    
                    if len(numeric_cols) < 3:
                        st.warning("Need at least 3 numeric columns")
                    else:
                        col1, col2, col3 = st.columns(3)
                        x_axis = col1.selectbox("X axis:", numeric_cols, key="3d_x")
                        y_axis = col2.selectbox("Y axis:", numeric_cols, index=1, key="3d_y")
                        z_axis = col3.selectbox("Z axis:", numeric_cols, index=2, key="3d_z")
                        
                        color_options = ["None"] + cat_cols + numeric_cols
                        color_by = st.selectbox("Color by:", color_options, key="3d_color")
                        
                        scatter_kwargs = {
                            "x": x_axis,
                            "y": y_axis,
                            "z": z_axis,
                            "title": f"3D: {x_axis}, {y_axis}, {z_axis}",
                            "opacity": 0.7
                        }
                        
                        if color_by != "None":
                            scatter_kwargs["color"] = color_by
                        else:
                            scatter_kwargs["color_discrete_sequence"] = ['#00A896']
                        
                        fig = px.scatter_3d(df, **scatter_kwargs)
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.caption("💡 Drag to rotate the 3D view!")
                
                # ----- Violin Plot -----
                elif plot_type == "Violin Plot":
                    st.markdown("Combines box plot with distribution shape")
                    
                    selected_col = st.selectbox(
                        "Select numeric column:",
                        numeric_cols,
                        key="violin_col"
                    )
                    
                    group_options = ["None"] + cat_cols
                    group_by = st.selectbox("Group by category:", group_options, key="violin_group")
                    
                    if group_by == "None":
                        fig = px.violin(
                            df,
                            y=selected_col,
                            box=True,
                            points="outliers",
                            title=f"Violin plot of {selected_col}",
                            color_discrete_sequence=['#00A896']
                        )
                    else:
                        fig = px.violin(
                            df,
                            x=group_by,
                            y=selected_col,
                            box=True,
                            points="outliers",
                            title=f"{selected_col} by {group_by}",
                            color=group_by
                        )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📖 How to read a violin plot"):
                        st.markdown("""
                        - **Width** at any point shows how many data points have that value
                        - **Box inside** shows the interquartile range (Q1 to Q3)
                        - **Line in box** is the median
                        - **Dots outside** are outliers
                        - Wider violin = more data points concentrated there
                        """)
                
                # ----- Categorical vs Numeric -----
                elif plot_type == "Categorical vs Numeric":
                    st.markdown("Compare numeric distributions across categories")
                    
                    if not cat_cols:
                        st.warning("No categorical columns found")
                    else:
                        col1, col2 = st.columns(2)
                        cat_col = col1.selectbox("Category column:", cat_cols, key="catnum_cat")
                        num_col = col2.selectbox("Numeric column:", numeric_cols, key="catnum_num")
                        
                        chart_type = st.radio(
                            "Chart type:",
                            [
                                "Box plot",
                                "Violin plot",
                                "Bar chart (mean)",
                                "Bar chart (sum)",
                                "Bar chart (count)",
                                "Strip plot",
                                "Swarm plot",
                                "Histogram (grouped)",
                                "Pie chart",
                                "Sunburst"
                            ],
                            horizontal=True,
                            key="catnum_chart"
                        )
                        
                        # Limit categories if too many
                        unique_cats = df[cat_col].nunique()
                        if unique_cats > 20:
                            st.warning(f"⚠️ {cat_col} has {unique_cats} unique values. Showing top 20 by frequency.")
                            top_cats = df[cat_col].value_counts().head(20).index
                            plot_df = df[df[cat_col].isin(top_cats)]
                        else:
                            plot_df = df
                        
                        # ----- Box plot -----
                        if chart_type == "Box plot":
                            fig = px.box(
                                plot_df, x=cat_col, y=num_col,
                                title=f"{num_col} by {cat_col}",
                                color=cat_col,
                                points="outliers"
                            )
                        
                        # ----- Violin plot -----
                        elif chart_type == "Violin plot":
                            fig = px.violin(
                                plot_df, x=cat_col, y=num_col,
                                title=f"{num_col} distribution by {cat_col}",
                                color=cat_col,
                                box=True,
                                points="outliers"
                            )
                        
                        # ----- Bar chart (mean) -----
                        elif chart_type == "Bar chart (mean)":
                            agg_df = plot_df.groupby(cat_col)[num_col].mean().reset_index()
                            agg_df = agg_df.sort_values(num_col, ascending=False)
                            fig = px.bar(
                                agg_df, x=cat_col, y=num_col,
                                title=f"Mean {num_col} by {cat_col}",
                                color=num_col,
                                color_continuous_scale="Viridis",
                                text_auto='.2f'
                            )
                        
                        # ----- Bar chart (sum) -----
                        elif chart_type == "Bar chart (sum)":
                            agg_df = plot_df.groupby(cat_col)[num_col].sum().reset_index()
                            agg_df = agg_df.sort_values(num_col, ascending=False)
                            fig = px.bar(
                                agg_df, x=cat_col, y=num_col,
                                title=f"Total {num_col} by {cat_col}",
                                color=num_col,
                                color_continuous_scale="Teal",
                                text_auto='.2s'
                            )
                        
                        # ----- Bar chart (count) -----
                        elif chart_type == "Bar chart (count)":
                            count_df = plot_df[cat_col].value_counts().reset_index()
                            count_df.columns = [cat_col, 'Count']
                            fig = px.bar(
                                count_df, x=cat_col, y='Count',
                                title=f"Count of {cat_col}",
                                color='Count',
                                color_continuous_scale="Blues",
                                text_auto=True
                            )
                        
                        # ----- Strip plot -----
                        elif chart_type == "Strip plot":
                            fig = px.strip(
                                plot_df, x=cat_col, y=num_col,
                                title=f"{num_col} by {cat_col} (all points)",
                                color=cat_col
                            )
                        
                        # ----- Swarm plot (using strip with jitter) -----
                        elif chart_type == "Swarm plot":
                            fig = px.strip(
                                plot_df, x=cat_col, y=num_col,
                                title=f"{num_col} swarm by {cat_col}",
                                color=cat_col
                            )
                            fig.update_traces(jitter=0.5, marker=dict(size=8, opacity=0.6))
                        
                        # ----- Histogram (grouped) -----
                        elif chart_type == "Histogram (grouped)":
                            fig = px.histogram(
                                plot_df, x=num_col,
                                color=cat_col,
                                title=f"Distribution of {num_col} by {cat_col}",
                                marginal="box",
                                opacity=0.7,
                                barmode="overlay"
                            )
                        
                        # ----- Pie chart -----
                        elif chart_type == "Pie chart":
                            agg_df = plot_df.groupby(cat_col)[num_col].sum().reset_index()
                            fig = px.pie(
                                agg_df, names=cat_col, values=num_col,
                                title=f"{num_col} share by {cat_col}",
                                hole=0.4
                            )
                        
                        # ----- Sunburst -----
                        elif chart_type == "Sunburst":
                            if len(cat_cols) >= 2:
                                second_cat = st.selectbox(
                                    "Second category for sunburst:",
                                    [c for c in cat_cols if c != cat_col],
                                    key="sunburst_second"
                                )
                                fig = px.sunburst(
                                    plot_df,
                                    path=[cat_col, second_cat],
                                    values=num_col,
                                    title=f"{num_col} by {cat_col} → {second_cat}"
                                )
                            else:
                                st.warning("Need at least 2 categorical columns for sunburst")
                                fig = None
                        
                        if fig is not None:
                            fig.update_layout(height=550)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show summary statistics
                        st.markdown("#### Summary by category")
                        summary = plot_df.groupby(cat_col)[num_col].agg([
                            ('Count', 'count'),
                            ('Mean', 'mean'),
                            ('Median', 'median'),
                            ('Std', 'std'),
                            ('Min', 'min'),
                            ('Max', 'max'),
                            ('Sum', 'sum')
                        ]).round(2).reset_index()
                        st.dataframe(summary, use_container_width=True, hide_index=True)
                
                # ----- Density Plot -----
                elif plot_type == "Density Plot":
                    st.markdown("Smoothed distribution curve (KDE)")
                    
                    selected_cols = st.multiselect(
                        "Select numeric columns:",
                        numeric_cols,
                        default=[numeric_cols[0]] if numeric_cols else [],
                        key="density_cols"
                    )
                    
                    if selected_cols:
                        fig = go.Figure()
                        colors = ['#00A896', '#1E2761', '#F96167', '#F5B942', '#7F77DD']
                        
                        for i, col in enumerate(selected_cols):
                            data = df[col].dropna()
                            
                            from scipy.stats import gaussian_kde
                            if len(data) > 1:
                                kde = gaussian_kde(data)
                                x_range = np.linspace(data.min(), data.max(), 200)
                                y_kde = kde(x_range)
                                
                                fig.add_trace(go.Scatter(
                                    x=x_range,
                                    y=y_kde,
                                    mode='lines',
                                    name=col,
                                    fill='tozeroy',
                                    line=dict(color=colors[i % len(colors)], width=2),
                                    opacity=0.6
                                ))
                        
                        fig.update_layout(
                            title="Density distributions",
                            xaxis_title="Value",
                            yaxis_title="Density",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ----- Parallel Coordinates -----
                elif plot_type == "Parallel Coordinates":
                    st.markdown("Visualize patterns across multiple dimensions")
                    
                    selected_cols = st.multiselect(
                        "Select numeric columns (3+ recommended):",
                        numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))],
                        key="parallel_cols"
                    )
                    
                    color_options = ["None"] + numeric_cols
                    color_by = st.selectbox("Color by:", color_options, key="parallel_color")
                    
                    if len(selected_cols) >= 2:
                        sample_size = min(500, len(df))
                        plot_df = df[selected_cols].sample(n=sample_size) if len(df) > sample_size else df[selected_cols]
                        
                        if color_by != "None" and color_by not in plot_df.columns:
                            plot_df = pd.concat([plot_df, df[color_by]], axis=1)
                        
                        kwargs = {"dimensions": selected_cols}
                        if color_by != "None":
                            kwargs["color"] = color_by
                            kwargs["color_continuous_scale"] = "Viridis"
                        
                        fig = px.parallel_coordinates(plot_df, **kwargs)
                        fig.update_layout(height=500, title="Parallel Coordinates Plot")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if len(df) > 500:
                            st.caption(f"Showing 500 sampled rows for clarity")
                        
                        with st.expander("📖 How to read parallel coordinates"):
                            st.markdown("""
                            Each line represents one row of your data, crossing through all dimensions.
                            - **Patterns** in lines reveal clusters and groupings
                            - **Crossings** between axes suggest negative correlation
                            - **Parallel lines** suggest positive correlation
                            - **Color** highlights groups or values
                            """)

# ============================================================
    # TAB 3: Outlier Detection
    # ============================================================
    with tab_outliers:
        from modules.outliers import (
            detect_outliers_iqr,
            detect_outliers_zscore,
            detect_outliers_modified_zscore,
            detect_outliers_isolation_forest,
            detect_outliers_lof,
            compare_outlier_methods
        )
        
        st.subheader("⚠️ Outlier Detection")
        st.markdown("Find unusual values that may need attention or removal")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for outlier detection")
        else:
            # Sub-tabs for different methods
            out_tab1, out_tab2, out_tab3 = st.tabs([
                "🔍 Single Column Analysis",
                "📊 Method Comparison",
                "🤖 Multivariate (ML-based)"
            ])
            
            # ----- Sub-tab 1: Single Column Analysis -----
            with out_tab1:
                st.markdown("### Detect outliers in one column")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_col = st.selectbox(
                        "Select numeric column:",
                        numeric_cols,
                        key="out_col"
                    )
                
                with col2:
                    method = st.selectbox(
                        "Detection method:",
                        ["IQR", "Z-score", "Modified Z-score"],
                        key="out_method"
                    )
                
                # Method parameters
                if method == "IQR":
                    multiplier = st.slider("IQR multiplier:", 1.0, 3.0, 1.5, 0.1)
                    result = detect_outliers_iqr(df[selected_col], multiplier=multiplier)
                elif method == "Z-score":
                    threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1)
                    result = detect_outliers_zscore(df[selected_col], threshold=threshold)
                else:
                    threshold = st.slider("Modified Z-score threshold:", 2.0, 5.0, 3.5, 0.1)
                    result = detect_outliers_modified_zscore(df[selected_col], threshold=threshold)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Show metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Outliers found", result["outlier_count"])
                    m2.metric("Percentage", f"{result['outlier_percentage']}%")
                    m3.metric("Total rows", len(df[selected_col].dropna()))
                    
                    # Method-specific details
                    if method == "IQR":
                        st.info(
                            f"**Lower bound:** {result['lower_bound']} | "
                            f"**Upper bound:** {result['upper_bound']} | "
                            f"**Q1:** {result['Q1']} | **Q3:** {result['Q3']} | **IQR:** {result['IQR']}"
                        )
                    elif method == "Z-score":
                        st.info(
                            f"**Mean:** {result['mean']} | "
                            f"**Std:** {result['std']} | "
                            f"**Threshold:** ±{result['threshold']} σ"
                        )
                    else:
                        st.info(
                            f"**Median:** {result['median']} | "
                            f"**MAD:** {result['MAD']} | "
                            f"**Threshold:** ±{result['threshold']}"
                        )
                    
                    st.markdown("---")
                    
                    # Visualization
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Box plot showing outliers
                        fig_box = px.box(
                            df, y=selected_col,
                            title=f"Box plot of {selected_col}",
                            points="outliers",
                            color_discrete_sequence=['#1E2761']
                        )
                        fig_box.update_layout(height=400)
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    with col_b:
                        # Scatter plot with outliers highlighted
                        plot_df = df[[selected_col]].copy().reset_index()
                        plot_df["is_outlier"] = result["outlier_mask"].reset_index(drop=True)
                        plot_df["color"] = plot_df["is_outlier"].map({True: "Outlier", False: "Normal"})
                        
                        fig_scatter = px.scatter(
                            plot_df,
                            x="index",
                            y=selected_col,
                            color="color",
                            color_discrete_map={"Normal": "#00A896", "Outlier": "#F96167"},
                            title=f"Outliers in {selected_col}",
                            labels={"index": "Row index"}
                        )
                        fig_scatter.update_layout(height=400)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Show outlier values
                    if result["outlier_count"] > 0:
                        st.markdown("#### Outlier values (first 50)")
                        outlier_values = result["outlier_values"]
                        outlier_df = pd.DataFrame({
                            "Index": range(len(outlier_values)),
                            "Value": outlier_values
                        })
                        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
                        
                        # Show full outlier rows
                        if st.checkbox("Show full rows containing outliers", key="show_outlier_rows"):
                            outlier_rows = df[result["outlier_mask"]]
                            st.dataframe(outlier_rows.head(50), use_container_width=True)
                            st.caption(f"Showing first 50 of {len(outlier_rows)} outlier rows")
            
            # ----- Sub-tab 2: Method Comparison -----
            with out_tab2:
                st.markdown("### Compare all outlier detection methods")
                st.markdown("See how different methods detect outliers in the same column")
                
                compare_col = st.selectbox(
                    "Select column to analyze:",
                    numeric_cols,
                    key="compare_col"
                )
                
                comparison = compare_outlier_methods(df[compare_col])
                
                if comparison.empty:
                    st.warning("Not enough data for comparison")
                else:
                    st.dataframe(comparison, use_container_width=True, hide_index=True)
                    
                    # Visualize comparison
                    fig_compare = px.bar(
                        comparison,
                        x="Method",
                        y="Outliers Found",
                        title=f"Outlier counts by method for {compare_col}",
                        color="Outliers Found",
                        color_continuous_scale="Reds",
                        text_auto=True
                    )
                    fig_compare.update_layout(height=400)
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    with st.expander("📖 Which method should I use?"):
                        st.markdown("""
                        - **IQR (1.5x):** Best general-purpose method. Good for most data.
                        - **Z-score:** Best when your data is approximately **normally distributed**.
                        - **Modified Z-score:** Best when data has **heavy outliers** (more robust).
                        
                        **Tips:**
                        - If methods disagree a lot, your data has unusual structure
                        - Always look at the box plot before deciding
                        - Don't blindly remove outliers — investigate first!
                        """)
            
            # ----- Sub-tab 3: Multivariate (ML-based) -----
            with out_tab3:
                st.markdown("### Multivariate outlier detection")
                st.markdown("Detect rows that are unusual across **multiple columns** at once")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_cols = st.multiselect(
                        "Select columns for analysis:",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))],
                        key="multi_cols"
                    )
                
                with col2:
                    ml_method = st.selectbox(
                        "Method:",
                        ["Isolation Forest", "Local Outlier Factor (LOF)"],
                        key="ml_method"
                    )
                
                contamination = st.slider(
                    "Expected outlier proportion (contamination):",
                    0.01, 0.5, 0.1, 0.01,
                    help="What % of your data do you expect to be outliers?"
                )
                
                if len(selected_cols) >= 2:
                    if st.button("🔍 Detect outliers", type="primary"):
                        with st.spinner(f"Running {ml_method}..."):
                            if ml_method == "Isolation Forest":
                                result = detect_outliers_isolation_forest(
                                    df, selected_cols, contamination=contamination
                                )
                            else:
                                result = detect_outliers_lof(
                                    df, selected_cols, contamination=contamination
                                )
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Metrics
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Outliers found", result["outlier_count"])
                            m2.metric("Percentage", f"{result['outlier_percentage']}%")
                            m3.metric("Method", result["method"])
                            
                            st.markdown("---")
                            
                            # Visualization (use first 2 columns for 2D plot)
                            if len(selected_cols) >= 2:
                                plot_df = df[selected_cols].copy().reset_index(drop=True)
                                plot_df["is_outlier"] = result["outlier_mask"].reset_index(drop=True)
                                plot_df["color"] = plot_df["is_outlier"].map({True: "Outlier", False: "Normal"})
                                
                                fig_2d = px.scatter(
                                    plot_df,
                                    x=selected_cols[0],
                                    y=selected_cols[1],
                                    color="color",
                                    color_discrete_map={"Normal": "#00A896", "Outlier": "#F96167"},
                                    title=f"Outliers detected by {result['method']}",
                                    opacity=0.6
                                )
                                fig_2d.update_layout(height=500)
                                st.plotly_chart(fig_2d, use_container_width=True)
                                
                                # 3D visualization if 3+ columns
                                if len(selected_cols) >= 3:
                                    fig_3d = px.scatter_3d(
                                        plot_df,
                                        x=selected_cols[0],
                                        y=selected_cols[1],
                                        z=selected_cols[2],
                                        color="color",
                                        color_discrete_map={"Normal": "#00A896", "Outlier": "#F96167"},
                                        title=f"3D outlier view",
                                        opacity=0.6
                                    )
                                    fig_3d.update_layout(height=500)
                                    st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Show outlier rows
                            if result["outlier_count"] > 0:
                                st.markdown("#### Outlier rows")
                                outlier_rows = df[result["outlier_mask"]][selected_cols + (
                                    [c for c in df.columns if c not in selected_cols][:3]
                                )]
                                st.dataframe(outlier_rows.head(50), use_container_width=True)
                                st.caption(f"Showing first 50 of {result['outlier_count']} outlier rows")
                else:
                    st.info("Select at least 2 columns to start")
                
                with st.expander("📖 ML-based vs Statistical methods"):
                    st.markdown("""
                    **Isolation Forest:**
                    - Works by randomly splitting data — outliers get isolated quickly
                    - Fast and scales well to large datasets
                    - Best for finding global outliers
                    
                    **Local Outlier Factor (LOF):**
                    - Compares local density of each point to its neighbors
                    - Better at finding local outliers in clustered data
                    - Slower but more accurate for complex distributions
                    
                    **When to use ML methods:**
                    - You have multiple correlated features
                    - Outliers are unusual combinations, not single extreme values
                    - Statistical methods miss what's "obviously wrong"
                    """)
                    
                    
# ============================================================
    # TAB 4: Preprocessing
    # ============================================================
    with tab_preprocessing:
        from modules.preprocessing import (
            handle_missing_values, drop_duplicates, encode_categorical,
            scale_features, drop_columns, convert_dtype, get_missing_summary
        )
        
        st.subheader("🛠️ Data Preprocessing")
        st.markdown("Clean and transform your data before modeling")
        
        # Info banner
        st.info(
            "💡 All changes are applied to a **working copy** of your data. "
            "Click 'Reset to original' to undo all changes."
        )
        
        # Initialize preprocessed df in session state
        if "df_processed" not in st.session_state or st.session_state.df_processed is None:
            st.session_state.df_processed = df.copy()
        
        working_df = st.session_state.df_processed
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Reset to original", use_container_width=True):
                st.session_state.df_processed = df.copy()
                st.rerun()
        with col2:
            if st.button("💾 Save changes", use_container_width=True, type="primary"):
                st.session_state.df = st.session_state.df_processed.copy()
                st.success("Changes saved to main dataset!")
        
        # Show current shape
        st.markdown(f"**Current shape:** {working_df.shape[0]:,} rows × {working_df.shape[1]} columns "
                    f"(original: {df.shape[0]:,} × {df.shape[1]})")
        
        st.markdown("---")
        
        # Sub-tabs for preprocessing steps
        pp_tab1, pp_tab2, pp_tab3, pp_tab4, pp_tab5 = st.tabs([
            "❓ Missing Values",
            "🔢 Encode Categories",
            "📏 Scale Features",
            "🗑️ Drop Columns",
            "🔄 Convert Types"
        ])
        
        # ----- Sub-tab 1: Missing Values -----
        with pp_tab1:
            st.markdown("### Handle missing values")
            
            missing_summary = get_missing_summary(working_df)
            
            if missing_summary.empty:
                st.success("✅ No missing values in the dataset!")
            else:
                st.markdown("#### Columns with missing values")
                st.dataframe(missing_summary, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("#### Fix missing values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_col = st.selectbox(
                        "Column:",
                        missing_summary["Column"].tolist(),
                        key="missing_col"
                    )
                
                with col2:
                    col_dtype = working_df[missing_col].dtype
                    is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
                    
                    if is_numeric:
                        methods = ["mean", "median", "mode", "zero", "forward_fill", "backward_fill", "custom", "drop"]
                    else:
                        methods = ["mode", "forward_fill", "backward_fill", "custom", "drop"]
                    
                    fill_method = st.selectbox(
                        "Method:",
                        methods,
                        key="fill_method",
                        help="drop = remove rows with missing values"
                    )
                
                custom_value = None
                if fill_method == "custom":
                    custom_value = st.text_input("Custom fill value:", key="custom_val")
                    if is_numeric and custom_value:
                        try:
                            custom_value = float(custom_value)
                        except:
                            st.warning("Invalid number")
                
                if st.button("✅ Apply", type="primary", key="apply_missing"):
                    before_count = working_df[missing_col].isna().sum()
                    working_df = handle_missing_values(working_df, missing_col, fill_method, custom_value)
                    st.session_state.df_processed = working_df
                    after_count = working_df[missing_col].isna().sum() if missing_col in working_df.columns else 0
                    st.success(f"Fixed {before_count - after_count} missing values in '{missing_col}'")
                    st.rerun()
                
                st.markdown("---")
                
                # Fill all numeric columns at once
                st.markdown("#### Quick fix all")
                quick_method = st.selectbox(
                    "Fill all numeric columns with:",
                    ["mean", "median", "zero"],
                    key="quick_method"
                )
                if st.button("🚀 Fix all numeric columns", key="quick_fix"):
                    numeric_cols = working_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if working_df[col].isna().any():
                            working_df = handle_missing_values(working_df, col, quick_method)
                    st.session_state.df_processed = working_df
                    st.success(f"Fixed all numeric columns with {quick_method}")
                    st.rerun()
        
        # ----- Sub-tab 2: Encode Categories -----
        with pp_tab2:
            st.markdown("### Encode categorical columns")
            st.markdown("Convert text/categories into numbers for ML models")
            
            cat_cols_list = working_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            if not cat_cols_list:
                st.info("No categorical columns to encode")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    encode_col = st.selectbox(
                        "Column to encode:",
                        cat_cols_list,
                        key="encode_col"
                    )
                    
                    # Show unique values
                    unique_count = working_df[encode_col].nunique()
                    st.caption(f"Unique values: {unique_count}")
                    
                    if unique_count <= 20:
                        with st.expander("View unique values"):
                            st.write(working_df[encode_col].value_counts())
                
                with col2:
                    encode_method = st.selectbox(
                        "Encoding method:",
                        ["label", "one_hot", "ordinal"],
                        key="encode_method",
                        help="Label: 0,1,2... | One-hot: binary columns | Ordinal: ordered integers"
                    )
                    
                    if encode_method == "label":
                        st.caption("Converts to integers (0, 1, 2, ...)")
                    elif encode_method == "one_hot":
                        st.caption(f"Creates {unique_count} new binary columns")
                    else:
                        st.caption("Preserves order if data is ordinal")
                
                if st.button("✅ Apply encoding", type="primary", key="apply_encode"):
                    if encode_method == "one_hot" and unique_count > 50:
                        st.error(f"Too many unique values ({unique_count}). One-hot would create too many columns.")
                    else:
                        working_df, info = encode_categorical(working_df, encode_col, encode_method)
                        st.session_state.df_processed = working_df
                        st.success(info)
                        st.rerun()
        
        # ----- Sub-tab 3: Scale Features -----
        with pp_tab3:
            st.markdown("### Scale numeric features")
            st.markdown("Normalize numeric columns for better ML model performance")
            
            numeric_cols_list = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols_list:
                st.info("No numeric columns to scale")
            else:
                scale_cols = st.multiselect(
                    "Columns to scale:",
                    numeric_cols_list,
                    default=numeric_cols_list,
                    key="scale_cols"
                )
                
                scale_method = st.radio(
                    "Scaling method:",
                    ["standard", "minmax", "robust"],
                    horizontal=True,
                    key="scale_method",
                    help="Standard: mean=0, std=1 | MinMax: 0 to 1 | Robust: uses median/IQR"
                )
                
                # Show what each method does
                if scale_method == "standard":
                    st.info("**Standard Scaling:** Transforms to mean=0, std=1. Best for algorithms like Linear Regression, Logistic Regression, SVM.")
                elif scale_method == "minmax":
                    st.info("**Min-Max Scaling:** Transforms to range [0, 1]. Best for Neural Networks and KNN.")
                else:
                    st.info("**Robust Scaling:** Uses median and IQR. Best when data has outliers.")
                
                # Preview before/after stats
                if scale_cols:
                    st.markdown("#### Current statistics")
                    preview_stats = working_df[scale_cols].describe().round(3)
                    st.dataframe(preview_stats, use_container_width=True)
                
                if st.button("✅ Apply scaling", type="primary", key="apply_scale"):
                    working_df, info = scale_features(working_df, scale_cols, scale_method)
                    st.session_state.df_processed = working_df
                    st.success(info)
                    st.rerun()
        
        # ----- Sub-tab 4: Drop Columns -----
        with pp_tab4:
            st.markdown("### Drop unwanted columns")
            
            cols_to_drop = st.multiselect(
                "Select columns to remove:",
                working_df.columns.tolist(),
                key="drop_cols"
            )
            
            if cols_to_drop:
                st.warning(f"⚠️ About to remove {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")
                
                if st.button("🗑️ Drop columns", type="primary", key="apply_drop"):
                    working_df = drop_columns(working_df, cols_to_drop)
                    st.session_state.df_processed = working_df
                    st.success(f"Dropped {len(cols_to_drop)} columns")
                    st.rerun()
            
            st.markdown("---")
            
            st.markdown("#### Remove duplicate rows")
            dup_count = working_df.duplicated().sum()
            
            if dup_count == 0:
                st.success("✅ No duplicate rows found")
            else:
                st.warning(f"Found {dup_count} duplicate rows")
                if st.button("🗑️ Remove duplicates", key="remove_dupes"):
                    working_df = drop_duplicates(working_df)
                    st.session_state.df_processed = working_df
                    st.success(f"Removed {dup_count} duplicate rows")
                    st.rerun()
        
        # ----- Sub-tab 5: Convert Types -----
        with pp_tab5:
            st.markdown("### Convert column data types")
            
            col1, col2 = st.columns(2)
            
            with col1:
                convert_col = st.selectbox(
                    "Column:",
                    working_df.columns.tolist(),
                    key="convert_col"
                )
                st.caption(f"Current type: `{working_df[convert_col].dtype}`")
            
            with col2:
                new_type = st.selectbox(
                    "New type:",
                    ["int", "float", "str", "category", "datetime"],
                    key="new_type"
                )
            
            if st.button("✅ Convert", type="primary", key="apply_convert"):
                working_df, info = convert_dtype(working_df, convert_col, new_type)
                st.session_state.df_processed = working_df
                if "Error" in info:
                    st.error(info)
                else:
                    st.success(info)
                    st.rerun()
        
        # ----- Preview of current state -----
        st.markdown("---")
        st.markdown("### 👁️ Current data preview")
        st.dataframe(working_df.head(10), use_container_width=True)
        
        
        # ============================================================
    # TAB 5: Feature Engineering
    # ============================================================
    with tab_fe:
        from modules.feature_engineering import (
            apply_math_transform, create_binned_feature, create_interaction,
            create_polynomial_features, extract_datetime_features,
            create_text_length_feature, create_word_count_feature
        )
        
        st.subheader("⚡ Feature Engineering")
        st.markdown("Create new features from existing ones to improve ML model performance")
        
        # Initialize if needed
        if "df_processed" not in st.session_state or st.session_state.df_processed is None:
            st.session_state.df_processed = df.copy()
        
        working_df = st.session_state.df_processed
        
        # Info banner
        st.info(
            f"💡 Working with {working_df.shape[0]:,} rows × {working_df.shape[1]} columns. "
            "New features will be added to your working dataset."
        )
        
        # Action buttons
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Save features", use_container_width=True, type="primary", key="save_fe"):
                st.session_state.df = st.session_state.df_processed.copy()
                st.success("Saved to main dataset!")
        
        st.markdown("---")
        
        # Sub-tabs for different feature engineering techniques
        fe_tab1, fe_tab2, fe_tab3, fe_tab4, fe_tab5, fe_tab6, fe_tab7 = st.tabs([
            "🔢 Math Transforms",
            "📦 Binning",
            "✖️ Interactions",
            "📐 Polynomial",
            "📅 Datetime",
            "📝 Text Features",
            "🎯 Feature Selection"
        ])
        
        # ----- Sub-tab 1: Math Transforms -----
        with fe_tab1:
            st.markdown("### Mathematical transformations")
            st.markdown("Apply log, sqrt, square, etc. to numeric columns")
            
            numeric_cols_list = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols_list:
                st.warning("No numeric columns available")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    transform_col = st.selectbox(
                        "Column:",
                        numeric_cols_list,
                        key="transform_col"
                    )
                
                with col2:
                    transform_type = st.selectbox(
                        "Transform:",
                        ["log", "log10", "sqrt", "square", "cube", "reciprocal", "exp", "abs"],
                        key="transform_type",
                        help="log: log(x+1) | sqrt: √x | square: x² | etc."
                    )
                
                # Show explanation
                explanations = {
                    "log": "Useful for right-skewed data (e.g., income, prices)",
                    "log10": "Log base 10 — useful for orders of magnitude",
                    "sqrt": "Reduces skewness, less aggressive than log",
                    "square": "Emphasizes differences between high values",
                    "cube": "Even more emphasis on high values",
                    "reciprocal": "1/x — useful for rates and ratios",
                    "exp": "Exponential — use carefully",
                    "abs": "Absolute value — removes negatives"
                }
                st.caption(f"💡 {explanations[transform_type]}")
                
                if st.button("✅ Create feature", type="primary", key="apply_transform"):
                    working_df, info = apply_math_transform(working_df, transform_col, transform_type)
                    st.session_state.df_processed = working_df
                    st.success(info)
                    st.rerun()
        
        # ----- Sub-tab 2: Binning -----
        with fe_tab2:
            st.markdown("### Convert numeric to categorical bins")
            st.markdown("Group continuous values into bins (e.g., ages → young/middle/old)")
            
            numeric_cols_list = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols_list:
                st.warning("No numeric columns available")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bin_col = st.selectbox(
                        "Column:",
                        numeric_cols_list,
                        key="bin_col"
                    )
                
                with col2:
                    n_bins = st.slider("Number of bins:", 2, 20, 5, key="n_bins")
                
                with col3:
                    bin_method = st.selectbox(
                        "Method:",
                        ["equal_width", "equal_frequency"],
                        key="bin_method",
                        help="Equal width: same range | Equal freq: same count"
                    )
                
                if st.button("✅ Create bins", type="primary", key="apply_bins"):
                    working_df, info = create_binned_feature(working_df, bin_col, n_bins, bin_method)
                    st.session_state.df_processed = working_df
                    st.success(info)
                    st.rerun()
        
        # ----- Sub-tab 3: Interactions -----
        with fe_tab3:
            st.markdown("### Create interactions between columns")
            st.markdown("Combine two columns (e.g., price × quantity = revenue)")
            
            numeric_cols_list = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols_list) < 2:
                st.warning("Need at least 2 numeric columns")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    int_col1 = st.selectbox(
                        "Column 1:",
                        numeric_cols_list,
                        key="int_col1"
                    )
                
                with col2:
                    int_col2 = st.selectbox(
                        "Column 2:",
                        [c for c in numeric_cols_list if c != int_col1],
                        key="int_col2"
                    )
                
                with col3:
                    operation = st.selectbox(
                        "Operation:",
                        ["multiply", "add", "subtract", "divide", "ratio"],
                        key="int_op"
                    )
                
                if st.button("✅ Create interaction", type="primary", key="apply_interaction"):
                    working_df, info = create_interaction(working_df, int_col1, int_col2, operation)
                    st.session_state.df_processed = working_df
                    st.success(info)
                    st.rerun()
        
        # ----- Sub-tab 4: Polynomial Features -----
        with fe_tab4:
            st.markdown("### Polynomial features")
            st.markdown("Create x², x×y, y² features automatically")
            
            numeric_cols_list = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols_list) < 1:
                st.warning("Need at least 1 numeric column")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    poly_cols = st.multiselect(
                        "Columns (2-4 recommended):",
                        numeric_cols_list,
                        default=numeric_cols_list[:min(2, len(numeric_cols_list))],
                        key="poly_cols"
                    )
                
                with col2:
                    degree = st.slider("Polynomial degree:", 2, 4, 2, key="poly_degree")
                
                st.warning(
                    f"⚠️ With {len(poly_cols)} columns and degree {degree}, "
                    f"this will create many new features. Start small."
                )
                
                if st.button("✅ Create polynomial features", type="primary", key="apply_poly"):
                    if len(poly_cols) == 0:
                        st.error("Select at least one column")
                    else:
                        working_df, info = create_polynomial_features(working_df, poly_cols, degree)
                        st.session_state.df_processed = working_df
                        st.success(info)
                        st.rerun()
        
        # ----- Sub-tab 5: Datetime Features -----
        with fe_tab5:
            st.markdown("### Extract features from datetime columns")
            
            all_cols = working_df.columns.tolist()
            
            if not all_cols:
                st.warning("No columns available")
            else:
                dt_col = st.selectbox(
                    "Column (datetime or string):",
                    all_cols,
                    key="dt_col"
                )
                
                dt_features = st.multiselect(
                    "Features to extract:",
                    ["year", "month", "day", "hour", "minute", "weekday", "week", "quarter", "is_weekend", "day_of_year"],
                    default=["year", "month", "day", "weekday"],
                    key="dt_features"
                )
                
                st.caption("💡 If the column isn't in datetime format, it will be auto-converted")
                
                if st.button("✅ Extract features", type="primary", key="apply_dt"):
                    if not dt_features:
                        st.error("Select at least one feature")
                    else:
                        working_df, info = extract_datetime_features(working_df, dt_col, dt_features)
                        st.session_state.df_processed = working_df
                        if "Error" in info:
                            st.error(info)
                        else:
                            st.success(info)
                            st.rerun()
        
        # ----- Sub-tab 6: Text Features -----
        with fe_tab6:
            st.markdown("### Extract features from text columns")
            st.markdown("Create length and word count features")
            
            text_cols = working_df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_cols:
                st.warning("No text columns available")
            else:
                text_col = st.selectbox(
                    "Text column:",
                    text_cols,
                    key="text_col"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Create character length feature", key="apply_text_len"):
                        working_df, info = create_text_length_feature(working_df, text_col)
                        st.session_state.df_processed = working_df
                        st.success(info)
                        st.rerun()
                
                with col2:
                    if st.button("✅ Create word count feature", key="apply_word_count"):
                        working_df, info = create_word_count_feature(working_df, text_col)
                        st.session_state.df_processed = working_df
                        st.success(info)
                        st.rerun()
        
        # ----- Sub-tab 7: Feature Selection -----
        with fe_tab7:
            from modules.ml_explainer import (
                select_features_univariate, select_features_mutual_info,
                select_features_rfe, remove_low_variance_features
            )
            from modules.ml_trainer import detect_problem_type
            
            st.markdown("### 🎯 Automatic Feature Selection")
            st.markdown("Find the most important features for your target variable")
            
            all_cols = working_df.columns.tolist()
            target_col_fs = st.selectbox("Target variable:", all_cols, key="fs_target")
            
            if target_col_fs:
                fs_problem = detect_problem_type(working_df[target_col_fs])
                
                available_features = [c for c in all_cols if c != target_col_fs]
                numeric_features = [c for c in available_features if pd.api.types.is_numeric_dtype(working_df[c])]
                
                if not numeric_features:
                    st.warning("No numeric features available. Encode categorical columns first.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        fs_method = st.selectbox(
                            "Method:",
                            ["Univariate (F-test)", "Mutual Information", "Recursive Feature Elimination", "Low Variance Filter"],
                            key="fs_method"
                        )
                    with c2:
                        k_features = st.slider("Number of features to select:", 1, len(numeric_features), min(10, len(numeric_features)))
                    
                    fs_features_all = st.multiselect(
                        "Features to evaluate:",
                        numeric_features,
                        default=numeric_features,
                        key="fs_features"
                    )
                    
                    if st.button("🔍 Run feature selection", type="primary", key="fs_run"):
                        valid_mask = working_df[fs_features_all + [target_col_fs]].notna().all(axis=1)
                        X = working_df.loc[valid_mask, fs_features_all]
                        y = working_df.loc[valid_mask, target_col_fs]
                        
                        if len(X) < 10:
                            st.error("Not enough valid data")
                        else:
                            with st.spinner(f"Running {fs_method}..."):
                                if fs_method == "Univariate (F-test)":
                                    result = select_features_univariate(X, y, fs_problem, k_features)
                                elif fs_method == "Mutual Information":
                                    result = select_features_mutual_info(X, y, fs_problem, k_features)
                                elif fs_method == "Recursive Feature Elimination":
                                    result = select_features_rfe(X, y, fs_problem, k_features)
                                else:
                                    result = remove_low_variance_features(X)
                            
                            if "error" in result.columns:
                                st.error(f"Failed: {result['error'].iloc[0]}")
                            else:
                                st.success("✅ Feature selection complete")
                                st.dataframe(result, use_container_width=True, hide_index=True)
                                
                                # Visualize
                                score_col = None
                                for col in ["Score", "MI Score", "Variance"]:
                                    if col in result.columns:
                                        score_col = col
                                        break
                                
                                if score_col:
                                    fig = px.bar(
                                        result.head(15),
                                        x=score_col,
                                        y="Feature",
                                        orientation='h',
                                        color=score_col,
                                        color_continuous_scale="Teal",
                                        title=f"Top 15 Features by {score_col}"
                                    )
                                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show selected features
                                if "Selected" in result.columns:
                                    selected = result[result["Selected"]]["Feature"].tolist()
                                    st.markdown("#### ✅ Recommended features")
                                    st.code(", ".join(selected))
        
        # ----- Preview of current state -----
        st.markdown("---")
        st.markdown("### 👁️ Current data preview")
        st.caption(f"Total columns: {len(working_df.columns)}")
        st.dataframe(working_df.head(10), use_container_width=True)
        
        # Show which columns are new
        if "df" in st.session_state:
            original_cols = set(st.session_state.df.columns)
            current_cols = set(working_df.columns)
            new_cols = current_cols - original_cols
            if new_cols:
                st.markdown("#### 🆕 New features created")
                st.write(list(new_cols))
        
        
    # ============================================================
    # TAB 6: ML Training
    # ============================================================
    # ============================================================
    # TAB 6: ML Training (with Auto-ML, CV, Tuning, History)
    # ============================================================
    with tab_ml:
        from modules.ml_trainer import (
            detect_problem_type, get_available_models, get_model,
            prepare_data, train_model, get_feature_importance
        )
        from modules.ml_evaluator import (
            evaluate_regression, evaluate_classification, get_confusion_matrix_df
        )
        from modules.ml_advanced import (
            run_auto_ml, run_cross_validation,
            get_hyperparameter_grid, run_hyperparameter_tuning
        )
        
        st.subheader("🤖 ML Training")
        
        # Initialize train history
        if "train_history" not in st.session_state:
            st.session_state.train_history = []
        
        # Use processed data if available
        train_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
        
        st.info(f"Training on {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
        
        # Sub-tabs
        ml_tab1, ml_tab2, ml_tab3, ml_tab4, ml_tab5, ml_tab6 = st.tabs([
            "🎯 Manual Training",
            "🚀 Auto-ML",
            "⚙️ Hyperparameter Tuning",
            "🔍 SHAP Explainer",
            "📈 ROC Curve",
            "📜 Train History"
        ])
        
        # ============================================================
        # SUB-TAB 1: Manual Training (existing logic)
        # ============================================================
        with ml_tab1:
            st.markdown("### Step 1: Select target variable (Y)")
            all_cols = train_df.columns.tolist()
            target_col = st.selectbox("Target:", all_cols, key="target_col")
            
            if target_col:
                problem_type = detect_problem_type(train_df[target_col])
                c1, c2, c3 = st.columns(3)
                c1.metric("Problem type", problem_type.capitalize())
                c2.metric("Unique values", train_df[target_col].nunique())
                c3.metric("Data type", str(train_df[target_col].dtype))
            
            st.markdown("### Step 2: Select features (X)")
            available_features = [c for c in all_cols if c != target_col]
            numeric_features = [c for c in available_features if pd.api.types.is_numeric_dtype(train_df[c])]
            
            feature_cols = st.multiselect(
                "Features:",
                available_features,
                default=numeric_features[:10] if numeric_features else [],
                key="feature_cols"
            )
            
            if feature_cols:
                non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
                if non_numeric:
                    st.warning(f"⚠️ Non-numeric features: **{', '.join(non_numeric)}**. Encode them in Preprocessing.")
            
            st.markdown("### Step 3: Model & settings")
            c1, c2, c3 = st.columns(3)
            with c1:
                if target_col:
                    available_models = get_available_models(problem_type)
                    selected_model = st.selectbox("Model:", available_models, key="model_name")
                else:
                    selected_model = None
            with c2:
                test_size = st.slider("Test size:", 0.1, 0.5, 0.2, 0.05, key="test_size")
            with c3:
                use_cv = st.checkbox("Use cross-validation", value=True, key="use_cv")
                if use_cv:
                    cv_folds = st.slider("CV folds:", 3, 10, 5, key="cv_folds")
            
            if st.button("🚀 Train model", type="primary", use_container_width=True, key="train_btn"):
                if not target_col:
                    st.error("Select a target")
                elif not feature_cols:
                    st.error("Select features")
                elif any(not pd.api.types.is_numeric_dtype(train_df[c]) for c in feature_cols):
                    st.error("Some features are not numeric. Encode them first.")
                else:
                    with st.spinner(f"Training {selected_model}..."):
                        data_prep = prepare_data(train_df, target_col, feature_cols, test_size)
                        if "error" in data_prep:
                            st.error(data_prep["error"])
                        else:
                            model = get_model(selected_model, problem_type)
                            result = train_model(
                                model,
                                data_prep["X_train"], data_prep["y_train"],
                                data_prep["X_test"], data_prep["y_test"],
                                problem_type
                            )
                            
                            training_record = {
                                "model": result["model"],
                                "model_name": selected_model,
                                "problem_type": problem_type,
                                "features": feature_cols,
                                "target": target_col,
                                "y_train": result["y_train"],
                                "y_test": result["y_test"],
                                "y_pred_train": result["y_pred_train"],
                                "y_pred_test": result["y_pred_test"],
                                "y_proba_test": result.get("y_proba_test"),
                                "train_time": result["train_time"],
                                "X_train": data_prep["X_train"],
                                "X_test": data_prep["X_test"],
                                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                            }
                            
                            # Cross-validation
                            if use_cv:
                                with st.spinner("Running cross-validation..."):
                                    cv_result = run_cross_validation(
                                        get_model(selected_model, problem_type),
                                        data_prep["X_train"], data_prep["y_train"],
                                        cv_folds, problem_type
                                    )
                                    training_record["cv_result"] = cv_result
                            
                            st.session_state.last_training = training_record
                            st.session_state.train_history.append(training_record)
                            st.success(f"✅ Trained in {result['train_time']}s!")
                            st.rerun()
            
            # Show results
            if "last_training" in st.session_state:
                st.markdown("---")
                st.markdown("## 📊 Training Results")
                training = st.session_state.last_training
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model", training["model_name"])
                c2.metric("Features", len(training["features"]))
                c3.metric("Training time", f"{training['train_time']}s")
                c4.metric("Type", training["problem_type"].capitalize())
                
                st.markdown("### 📏 Metrics")
                if training["problem_type"] == "regression":
                    metrics = evaluate_regression(
                        training["y_train"], training["y_pred_train"],
                        training["y_test"], training["y_pred_test"]
                    )
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("R² (Test)", metrics["R² (Test)"])
                    mc2.metric("RMSE", metrics["RMSE (Test)"])
                    mc3.metric("MAE", metrics["MAE (Test)"])
                    
                    metrics_df = pd.DataFrame([metrics]).T.reset_index()
                    metrics_df.columns = ["Metric", "Value"]
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Actual vs Predicted
                    pred_df = pd.DataFrame({"Actual": training["y_test"], "Predicted": training["y_pred_test"]})
                    fig = px.scatter(pred_df, x="Actual", y="Predicted", opacity=0.6,
                                     color_discrete_sequence=['#00A896'], title="Actual vs Predicted")
                    min_v = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
                    max_v = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
                    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                             mode='lines', line=dict(color='red', dash='dash'), name='Perfect'))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    metrics = evaluate_classification(
                        training["y_train"], training["y_pred_train"],
                        training["y_test"], training["y_pred_test"],
                        training.get("y_proba_test")
                    )
                    if "error" not in metrics:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Accuracy", metrics["Accuracy (Test)"])
                        mc2.metric("Precision", metrics["Precision (Test)"])
                        mc3.metric("Recall", metrics["Recall (Test)"])
                        mc4.metric("F1", metrics["F1 Score (Test)"])
                        
                        cm_df = get_confusion_matrix_df(training["y_test"], training["y_pred_test"])
                        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
                        fig_cm.update_layout(height=500)
                        st.plotly_chart(fig_cm, use_container_width=True)
                
                # Cross-validation results
                if "cv_result" in training and training["cv_result"]:
                    st.markdown("### 🔄 Cross-Validation Results")
                    cv_data = []
                    for metric_name, values in training["cv_result"].items():
                        cv_data.append({
                            "Metric": metric_name,
                            "Mean": values["mean"],
                            "Std": values["std"],
                            "Min": values["min"],
                            "Max": values["max"]
                        })
                    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)
                    st.caption(f"Results from {len(list(training['cv_result'].values())[0]['scores'])} folds")
                
                # Feature importance
                importance_df = get_feature_importance(training["model"], training["features"])
                if not importance_df.empty:
                    st.markdown("### 🎯 Feature Importance")
                    fig_imp = px.bar(importance_df.head(15), x="Importance", y="Feature",
                                     orientation='h', color="Importance", color_continuous_scale="Teal")
                    fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
        
        # ============================================================
        # SUB-TAB 2: Auto-ML
        # ============================================================
        with ml_tab2:
            st.markdown("### 🚀 Auto-ML: Train all models at once")
            st.markdown("Automatically trains all available models and ranks them by performance")
            
            auto_target = st.selectbox("Target variable:", train_df.columns.tolist(), key="auto_target")
            
            if auto_target:
                auto_problem = detect_problem_type(train_df[auto_target])
                st.caption(f"Detected: **{auto_problem}**")
            
            auto_available = [c for c in train_df.columns if c != auto_target]
            auto_numeric = [c for c in auto_available if pd.api.types.is_numeric_dtype(train_df[c])]
            auto_features = st.multiselect(
                "Features:",
                auto_available,
                default=auto_numeric[:10] if auto_numeric else [],
                key="auto_features"
            )
            
            c1, c2 = st.columns(2)
            auto_test_size = c1.slider("Test size:", 0.1, 0.5, 0.2, 0.05, key="auto_test_size")
            auto_cv_folds = c2.slider("CV folds:", 3, 10, 5, key="auto_cv_folds")
            
            if st.button("🚀 Run Auto-ML (trains all models)", type="primary", use_container_width=True, key="automl_btn"):
                if not auto_features:
                    st.error("Select features")
                elif any(not pd.api.types.is_numeric_dtype(train_df[c]) for c in auto_features):
                    st.error("Encode non-numeric features first")
                else:
                    with st.spinner("Training all models... This may take a minute ☕"):
                        data_prep = prepare_data(train_df, auto_target, auto_features, auto_test_size)
                        
                        if "error" in data_prep:
                            st.error(data_prep["error"])
                        else:
                            results_df = run_auto_ml(
                                data_prep["X_train"], data_prep["y_train"],
                                data_prep["X_test"], data_prep["y_test"],
                                auto_problem, auto_cv_folds
                            )
                            st.session_state.auto_ml_results = results_df
                            st.rerun()
            
            # Display Auto-ML results
            if "auto_ml_results" in st.session_state:
                results_df = st.session_state.auto_ml_results
                
                if not results_df.empty:
                    st.markdown("### 🏆 Model Leaderboard")
                    
                    # Best model callout
                    best = results_df.iloc[0]
                    main_metric = "R² (Test)" if auto_problem == "regression" else "Accuracy (Test)"
                    st.success(f"🥇 **Best model:** {best['Model']} with **{main_metric}** = {best[main_metric]}")
                    
                    # Display leaderboard (without _model column)
                    display_df = results_df.drop(columns=["_model"])
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    fig = px.bar(
                        display_df,
                        x="Model",
                        y=main_metric,
                        title=f"Model Comparison — {main_metric}",
                        color=main_metric,
                        color_continuous_scale="Viridis",
                        text_auto='.3f'
                    )
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time comparison
                    fig_time = px.bar(
                        display_df,
                        x="Model",
                        y="Time (s)",
                        title="Training Time Comparison",
                        color="Time (s)",
                        color_continuous_scale="Reds",
                        text_auto='.2f'
                    )
                    fig_time.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_time, use_container_width=True)
                    
                    # Select best for further use
                    st.markdown("#### 💾 Use this model")
                    if st.button(f"Use {best['Model']} as the current model", key="use_best"):
                        data_prep = prepare_data(train_df, auto_target, auto_features, auto_test_size)
                        model = best["_model"]
                        y_pred_train = model.predict(data_prep["X_train"])
                        y_pred_test = model.predict(data_prep["X_test"])
                        
                        st.session_state.last_training = {
                            "model": model,
                            "model_name": best["Model"],
                            "problem_type": auto_problem,
                            "features": auto_features,
                            "target": auto_target,
                            "y_train": data_prep["y_train"],
                            "y_test": data_prep["y_test"],
                            "y_pred_train": y_pred_train,
                            "y_pred_test": y_pred_test,
                            "train_time": best["Time (s)"],
                            "X_train": data_prep["X_train"],
                            "X_test": data_prep["X_test"],
                            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                        }
                        st.session_state.train_history.append(st.session_state.last_training)
                        st.success(f"✅ {best['Model']} is now your active model!")
                        st.rerun()
        
        # ============================================================
        # SUB-TAB 3: Hyperparameter Tuning
        # ============================================================
        with ml_tab3:
            st.markdown("### ⚙️ Hyperparameter Tuning (GridSearchCV)")
            st.markdown("Automatically find the best model parameters")
            
            tune_target = st.selectbox("Target:", train_df.columns.tolist(), key="tune_target")
            
            if tune_target:
                tune_problem = detect_problem_type(train_df[tune_target])
            
            tune_available = [c for c in train_df.columns if c != tune_target]
            tune_numeric = [c for c in tune_available if pd.api.types.is_numeric_dtype(train_df[c])]
            tune_features = st.multiselect(
                "Features:",
                tune_available,
                default=tune_numeric[:10] if tune_numeric else [],
                key="tune_features"
            )
            
            if tune_target:
                tune_models = get_available_models(tune_problem)
                tune_model_name = st.selectbox("Model to tune:", tune_models, key="tune_model")
                
                # Show the parameter grid
                param_grid = get_hyperparameter_grid(tune_model_name, tune_problem)
                if param_grid:
                    with st.expander("View parameter grid"):
                        total_combos = 1
                        for k, v in param_grid.items():
                            st.write(f"**{k}:** {v}")
                            total_combos *= len(v)
                        st.info(f"Will test {total_combos} parameter combinations")
                else:
                    st.warning(f"No tuning grid available for {tune_model_name}. Try Random Forest, Gradient Boosting, or KNN.")
            
            tune_cv = st.slider("CV folds:", 3, 10, 5, key="tune_cv")
            tune_test_size = st.slider("Test size:", 0.1, 0.5, 0.2, 0.05, key="tune_test_size")
            
            if st.button("⚙️ Run hyperparameter tuning", type="primary", use_container_width=True, key="tune_btn"):
                if not tune_features:
                    st.error("Select features")
                elif not param_grid:
                    st.error("No parameter grid available for this model")
                else:
                    with st.spinner("Searching for best parameters... ⚙️"):
                        data_prep = prepare_data(train_df, tune_target, tune_features, tune_test_size)
                        
                        if "error" in data_prep:
                            st.error(data_prep["error"])
                        else:
                            base_model = get_model(tune_model_name, tune_problem)
                            tune_result = run_hyperparameter_tuning(
                                base_model, data_prep["X_train"], data_prep["y_train"],
                                param_grid, tune_cv, tune_problem
                            )
                            
                            if "error" in tune_result:
                                st.error(tune_result["error"])
                            else:
                                st.success(f"✅ Tested {tune_result['total_combinations']} combinations in {tune_result['search_time']}s")
                                
                                c1, c2 = st.columns(2)
                                c1.metric("Best CV Score", tune_result["best_score"])
                                c2.metric("Search time", f"{tune_result['search_time']}s")
                                
                                st.markdown("#### 🏆 Best parameters")
                                st.json(tune_result["best_params"])
                                
                                st.markdown("#### 📊 Top 5 combinations")
                                st.dataframe(tune_result["top_5"], use_container_width=True, hide_index=True)
                                
                                # Use best model
                                if st.button("💾 Use best model as active", key="use_tuned"):
                                    best_model = tune_result["best_model"]
                                    y_pred_train = best_model.predict(data_prep["X_train"])
                                    y_pred_test = best_model.predict(data_prep["X_test"])
                                    
                                    st.session_state.last_training = {
                                        "model": best_model,
                                        "model_name": f"{tune_model_name} (tuned)",
                                        "problem_type": tune_problem,
                                        "features": tune_features,
                                        "target": tune_target,
                                        "y_train": data_prep["y_train"],
                                        "y_test": data_prep["y_test"],
                                        "y_pred_train": y_pred_train,
                                        "y_pred_test": y_pred_test,
                                        "train_time": tune_result["search_time"],
                                        "X_train": data_prep["X_train"],
                                        "X_test": data_prep["X_test"],
                                        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                                    }
                                    st.session_state.train_history.append(st.session_state.last_training)
                                    st.success("✅ Tuned model is now active!")
                                    st.rerun()
        
        # ============================================================
        # SUB-TAB 4: SHAP Explainer
        # ============================================================
        with ml_tab4:
            from modules.ml_explainer import get_shap_values
            
            st.markdown("### 🔍 SHAP Explanations")
            st.markdown("Understand how each feature impacts predictions")
            
            if "last_training" not in st.session_state:
                st.warning("⚠️ Train a model first in the Manual Training tab")
            else:
                training = st.session_state.last_training
                
                st.info(f"Explaining: **{training['model_name']}** on **{training['target']}**")
                
                max_samples = st.slider(
                    "Samples to analyze:",
                    50, 500, 100, 50,
                    help="More samples = more accurate but slower",
                    key="shap_samples"
                )
                
                if st.button("🔍 Compute SHAP values", type="primary", key="shap_btn"):
                    with st.spinner("Computing SHAP values... (this can take a minute for non-tree models)"):
                        shap_result = get_shap_values(
                            training["model"],
                            training["X_test"],
                            max_samples=max_samples
                        )
                    
                    if "error" in shap_result:
                        st.error(f"SHAP failed: {shap_result['error']}")
                    else:
                        st.session_state.shap_result = shap_result
                        st.success(f"✅ SHAP values computed using {shap_result['explainer_type']}")
                        st.rerun()
                
                # Show SHAP results
                if "shap_result" in st.session_state:
                    shap_result = st.session_state.shap_result
                    
                    st.markdown("---")
                    
                    # Feature importance
                    st.markdown("#### 🎯 SHAP Feature Importance")
                    st.caption("Mean absolute SHAP value = average impact on predictions")
                    
                    importance_df = shap_result["importance_df"]
                    
                    fig_imp = px.bar(
                        importance_df.head(15),
                        x="SHAP Importance",
                        y="Feature",
                        orientation='h',
                        color="SHAP Importance",
                        color_continuous_scale="Teal",
                        title="Top 15 Features by SHAP Importance"
                    )
                    fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)
                    
                    # SHAP value distribution per feature
                    st.markdown("#### 📊 SHAP Value Distribution")
                    st.caption("How each feature's impact varies across samples")
                    
                    top_features = importance_df.head(5)["Feature"].tolist()
                    selected_feat = st.selectbox(
                        "Select feature to analyze:",
                        top_features,
                        key="shap_feat"
                    )
                    
                    if selected_feat:
                        feat_idx = list(shap_result["X_sample"].columns).index(selected_feat)
                        shap_vals_for_feature = shap_result["shap_values"][:, feat_idx]
                        feature_values = shap_result["X_sample"][selected_feat].values
                        
                        # Scatter plot: feature value vs SHAP value
                        scatter_df = pd.DataFrame({
                            "Feature Value": feature_values,
                            "SHAP Value": shap_vals_for_feature
                        })
                        
                        fig_scatter = px.scatter(
                            scatter_df,
                            x="Feature Value",
                            y="SHAP Value",
                            title=f"How {selected_feat} affects predictions",
                            color="SHAP Value",
                            color_continuous_scale="RdBu_r",
                            opacity=0.7
                        )
                        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_scatter.update_layout(height=500)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        st.caption(
                            f"**Positive SHAP value** = pushes prediction higher. "
                            f"**Negative SHAP value** = pushes prediction lower."
                        )
                    
                    # Single prediction explanation
                    st.markdown("#### 🎯 Explain a single prediction")
                    
                    sample_idx = st.number_input(
                        "Sample index to explain:",
                        0, len(shap_result["X_sample"]) - 1, 0,
                        key="shap_sample_idx"
                    )
                    
                    if sample_idx is not None:
                        sample_values = shap_result["X_sample"].iloc[sample_idx]
                        sample_shap = shap_result["shap_values"][sample_idx]
                        
                        explanation_df = pd.DataFrame({
                            "Feature": shap_result["X_sample"].columns,
                            "Value": sample_values.values,
                            "SHAP Impact": sample_shap
                        }).sort_values("SHAP Impact", key=abs, ascending=False)
                        
                        explanation_df["SHAP Impact"] = explanation_df["SHAP Impact"].round(4)
                        
                        fig_waterfall = px.bar(
                            explanation_df.head(10),
                            x="SHAP Impact",
                            y="Feature",
                            orientation='h',
                            color="SHAP Impact",
                            color_continuous_scale="RdBu_r",
                            title=f"Top feature impacts for sample #{sample_idx}",
                            text="Value"
                        )
                        fig_waterfall.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                        st.dataframe(explanation_df, use_container_width=True, hide_index=True)
        
        # ============================================================
        # SUB-TAB 5: ROC Curve
        # ============================================================
        with ml_tab5:
            from modules.ml_explainer import get_roc_data
            
            st.markdown("### 📈 ROC Curve (Classification only)")
            st.markdown("Shows the tradeoff between true positive rate and false positive rate")
            
            if "last_training" not in st.session_state:
                st.warning("⚠️ Train a classification model first")
            else:
                training = st.session_state.last_training
                
                if training["problem_type"] != "classification":
                    st.warning("⚠️ ROC curves only apply to classification models. Train a classifier first.")
                elif training.get("y_proba_test") is None:
                    st.warning("⚠️ This model doesn't provide probability predictions")
                else:
                    st.info(f"ROC for: **{training['model_name']}** on **{training['target']}**")
                    
                    classes = training["model"].classes_
                    roc_data = get_roc_data(
                        training["y_test"],
                        training["y_proba_test"],
                        list(classes)
                    )
                    
                    if "error" in roc_data:
                        st.error(f"Failed: {roc_data['error']}")
                    else:
                        if roc_data["binary"]:
                            # Binary ROC
                            st.metric("AUC Score", roc_data["auc"])
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=roc_data["fpr"],
                                y=roc_data["tpr"],
                                mode='lines',
                                name=f'ROC (AUC = {roc_data["auc"]})',
                                line=dict(color='#00A896', width=3),
                                fill='tozeroy'
                            ))
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random classifier',
                                line=dict(color='gray', dash='dash')
                            ))
                            fig.update_layout(
                                title="ROC Curve",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                height=500,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            auc_val = roc_data["auc"]
                            if auc_val >= 0.9:
                                st.success(f"🌟 Excellent classifier (AUC = {auc_val})")
                            elif auc_val >= 0.8:
                                st.success(f"✅ Good classifier (AUC = {auc_val})")
                            elif auc_val >= 0.7:
                                st.info(f"👍 Fair classifier (AUC = {auc_val})")
                            elif auc_val >= 0.6:
                                st.warning(f"⚠️ Weak classifier (AUC = {auc_val})")
                            else:
                                st.error(f"❌ Poor classifier (AUC = {auc_val})")
                        
                        else:
                            # Multi-class ROC
                            st.markdown("#### Per-class ROC curves")
                            
                            fig = go.Figure()
                            colors = ['#00A896', '#1E2761', '#F96167', '#F5B942', '#7F77DD']
                            
                            for i, (class_name, data) in enumerate(roc_data["classes_data"].items()):
                                fig.add_trace(go.Scatter(
                                    x=data["fpr"],
                                    y=data["tpr"],
                                    mode='lines',
                                    name=f'{class_name} (AUC = {data["auc"]})',
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ))
                            
                            fig.add_trace(go.Scatter(
                                x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(color='gray', dash='dash')
                            ))
                            fig.update_layout(
                                title="Multi-class ROC",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # AUC table
                            auc_df = pd.DataFrame([
                                {"Class": name, "AUC": data["auc"]}
                                for name, data in roc_data["classes_data"].items()
                            ])
                            st.dataframe(auc_df, use_container_width=True, hide_index=True)
                    
                    with st.expander("📖 How to read ROC curves"):
                        st.markdown("""
                        - **Top-left corner** = perfect classifier
                        - **Diagonal line** = random guessing (AUC = 0.5)
                        - **Higher AUC** = better separation between classes
                        
                        **AUC interpretation:**
                        - 0.9-1.0: Excellent
                        - 0.8-0.9: Good
                        - 0.7-0.8: Fair
                        - 0.6-0.7: Poor
                        - 0.5-0.6: Fail
                        """)
                        
        # ============================================================
        # SUB-TAB 6: Train History
        # ============================================================
        with ml_tab6:
            st.markdown("### 📜 Training History")
            st.markdown("Compare all models you've trained in this session")
            
            history = st.session_state.train_history
            
            if not history:
                st.info("No models trained yet. Train some models in the other tabs!")
            else:
                st.markdown(f"**{len(history)}** model(s) trained this session")
                
                # Build history DataFrame
                history_data = []
                for i, h in enumerate(history):
                    row = {
                        "#": i + 1,
                        "Time": h.get("timestamp", "?"),
                        "Model": h["model_name"],
                        "Target": h["target"],
                        "Features": len(h["features"]),
                        "Type": h["problem_type"],
                        "Train Time (s)": h.get("train_time", "?"),
                    }
                    
                    if h["problem_type"] == "regression":
                        from sklearn.metrics import r2_score
                        row["R² (Test)"] = round(r2_score(h["y_test"], h["y_pred_test"]), 4)
                    else:
                        from sklearn.metrics import accuracy_score
                        row["Accuracy (Test)"] = round(accuracy_score(h["y_test"], h["y_pred_test"]), 4)
                    
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                
                # Visualize
                if len(history) > 1:
                    metric_col = "R² (Test)" if "R² (Test)" in history_df.columns else "Accuracy (Test)"
                    if metric_col in history_df.columns:
                        fig = px.bar(
                            history_df, x="Model", y=metric_col,
                            title=f"Model Performance Across Session — {metric_col}",
                            color=metric_col, color_continuous_scale="Teal",
                            text_auto='.3f'
                        )
                        fig.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🗑️ Clear history", key="clear_history"):
                    st.session_state.train_history = []
                    st.rerun()
            
 # ============================================================
    # TAB 7: Predictions
    # ============================================================
    with tab_predict:
        st.subheader("🔮 Make Predictions")
        st.markdown("Use your trained model to predict on new data")
        
        if "last_training" not in st.session_state:
            st.warning("⚠️ No trained model available. Go to **ML Training** tab first to train a model.")
        else:
            training = st.session_state.last_training
            model = training["model"]
            features = training["features"]
            target = training["target"]
            problem_type = training["problem_type"]
            
            # Show model info
            st.info(
                f"**Current model:** {training['model_name']} | "
                f"**Target:** {target} | "
                f"**Features:** {len(features)} | "
                f"**Type:** {problem_type.capitalize()}"
            )
            
            st.markdown("---")
            
            # Two prediction modes
            mode = st.radio(
                "Prediction mode:",
                ["Manual input (single prediction)", "Upload file (batch predictions)", "Predict on current dataset"],
                horizontal=True,
                key="predict_mode"
            )
            
            # ----- Mode 1: Manual Input -----
            if mode == "Manual input (single prediction)":
                st.markdown("### Enter feature values")
                
                input_values = {}
                
                # Create input fields for each feature
                # Show 3 inputs per row
                cols_per_row = 3
                feature_chunks = [features[i:i + cols_per_row] for i in range(0, len(features), cols_per_row)]
                
                for chunk in feature_chunks:
                    cols = st.columns(cols_per_row)
                    for i, feat in enumerate(chunk):
                        with cols[i]:
                            # Get sample stats from training data
                            sample_data = training["X_train"][feat]
                            mean_val = float(sample_data.mean())
                            min_val = float(sample_data.min())
                            max_val = float(sample_data.max())
                            
                            input_values[feat] = st.number_input(
                                f"**{feat}**",
                                value=round(mean_val, 3),
                                help=f"Range: {round(min_val, 2)} to {round(max_val, 2)} | Mean: {round(mean_val, 2)}",
                                key=f"pred_input_{feat}"
                            )
                
                st.markdown("---")
                
                if st.button("🔮 Predict", type="primary", use_container_width=True, key="manual_predict"):
                    try:
                        # Create input DataFrame
                        input_df = pd.DataFrame([input_values])
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        
                        # Display result
                        st.markdown("### 🎯 Prediction Result")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"#### Predicted {target}:")
                            if problem_type == "regression":
                                st.markdown(f"# `{round(float(prediction), 4)}`")
                            else:
                                st.markdown(f"# `{prediction}`")
                        
                        with col2:
                            # For classification, show probabilities
                            if problem_type == "classification" and hasattr(model, "predict_proba"):
                                proba = model.predict_proba(input_df)[0]
                                classes = model.classes_
                                
                                proba_df = pd.DataFrame({
                                    "Class": classes,
                                    "Probability": proba
                                }).sort_values("Probability", ascending=False)
                                
                                st.markdown("#### Confidence scores:")
                                fig_proba = px.bar(
                                    proba_df, x="Probability", y="Class",
                                    orientation='h',
                                    color="Probability",
                                    color_continuous_scale="Teal",
                                    text_auto='.2%'
                                )
                                fig_proba.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig_proba, use_container_width=True)
                        
                        # Show the input values used
                        with st.expander("View input values used"):
                            st.dataframe(input_df, use_container_width=True, hide_index=True)
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            
            # ----- Mode 2: Upload File -----
            elif mode == "Upload file (batch predictions)":
                st.markdown("### Upload file with new data")
                st.caption("File must have these columns: " + ", ".join(features))
                
                pred_file = st.file_uploader(
                    "Upload CSV, Excel, or JSON with new data",
                    type=["csv", "xlsx", "xls", "json"],
                    key="pred_file_upload"
                )
                
                if pred_file is not None:
                    # Load the file
                    from modules.data_loader import load_file
                    new_df, error = load_file(pred_file)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Loaded {new_df.shape[0]} rows × {new_df.shape[1]} columns")
                        
                        # Check if all required features are present
                        missing_features = [f for f in features if f not in new_df.columns]
                        
                        if missing_features:
                            st.error(f"Missing required features: {missing_features}")
                            st.markdown("**Your file has these columns:**")
                            st.write(list(new_df.columns))
                        else:
                            # Select only the features needed
                            input_df = new_df[features].copy()
                            
                            # Show preview
                            st.markdown("#### Input data preview")
                            st.dataframe(input_df.head(10), use_container_width=True)
                            
                            # Check for missing values
                            missing_count = input_df.isna().sum().sum()
                            if missing_count > 0:
                                st.warning(f"⚠️ Found {missing_count} missing values. Rows with missing values will be skipped.")
                            
                            if st.button("🔮 Generate predictions", type="primary", key="batch_predict"):
                                try:
                                    # Drop rows with missing values
                                    valid_df = input_df.dropna()
                                    
                                    if len(valid_df) == 0:
                                        st.error("No valid rows to predict (all have missing values)")
                                    else:
                                        # Predict
                                        predictions = model.predict(valid_df)
                                        
                                        # Create results DataFrame
                                        result_df = new_df.loc[valid_df.index].copy()
                                        result_df[f"predicted_{target}"] = predictions
                                        
                                        # Add probabilities for classification
                                        if problem_type == "classification" and hasattr(model, "predict_proba"):
                                            try:
                                                probas = model.predict_proba(valid_df)
                                                for i, class_name in enumerate(model.classes_):
                                                    result_df[f"prob_{class_name}"] = probas[:, i]
                                            except:
                                                pass
                                        
                                        st.success(f"✅ Generated {len(predictions)} predictions!")
                                        
                                        # Show results
                                        st.markdown("### 📊 Predictions")
                                        st.dataframe(result_df, use_container_width=True)
                                        
                                        # Summary stats
                                        if problem_type == "regression":
                                            st.markdown("#### Prediction statistics")
                                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                                            stat_col1.metric("Mean", round(predictions.mean(), 3))
                                            stat_col2.metric("Min", round(predictions.min(), 3))
                                            stat_col3.metric("Max", round(predictions.max(), 3))
                                            
                                            # Histogram of predictions
                                            fig_hist = px.histogram(
                                                x=predictions,
                                                nbins=30,
                                                title="Distribution of predictions",
                                                color_discrete_sequence=['#00A896'],
                                                labels={'x': f'Predicted {target}'}
                                            )
                                            fig_hist.update_layout(height=400)
                                            st.plotly_chart(fig_hist, use_container_width=True)
                                        
                                        else:  # Classification
                                            st.markdown("#### Prediction distribution")
                                            pred_counts = pd.Series(predictions).value_counts().reset_index()
                                            pred_counts.columns = ['Class', 'Count']
                                            
                                            fig_bar = px.bar(
                                                pred_counts, x='Class', y='Count',
                                                title="Predictions by class",
                                                color='Count',
                                                color_continuous_scale="Teal",
                                                text_auto=True
                                            )
                                            fig_bar.update_layout(height=400)
                                            st.plotly_chart(fig_bar, use_container_width=True)
                                        
                                        # Download results
                                        csv = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download predictions as CSV",
                                            data=csv,
                                            file_name=f"predictions_{target}.csv",
                                            mime="text/csv",
                                            type="primary"
                                        )
                                
                                except Exception as e:
                                    st.error(f"Prediction failed: {str(e)}")
            
            # ----- Mode 3: Predict on current dataset -----
            else:
                st.markdown("### Predict on current dataset")
                st.caption("This will apply the model to the currently loaded dataset")
                
                current_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
                
                # Check features
                missing_features = [f for f in features if f not in current_df.columns]
                
                if missing_features:
                    st.error(f"Current dataset is missing these features: {missing_features}")
                else:
                    input_df = current_df[features].copy()
                    valid_df = input_df.dropna()
                    
                    st.info(f"Will predict on {len(valid_df):,} rows (of {len(current_df):,} total)")
                    
                    if st.button("🔮 Predict on full dataset", type="primary", key="dataset_predict"):
                        try:
                            predictions = model.predict(valid_df)
                            
                            # Build result
                            result_df = current_df.loc[valid_df.index].copy()
                            result_df[f"predicted_{target}"] = predictions
                            
                            # Compare with actual if target exists
                            if target in current_df.columns:
                                actual = current_df.loc[valid_df.index, target]
                                
                                st.markdown("### 📊 Predictions vs Actuals")
                                
                                if problem_type == "regression":
                                    # Compute R² and RMSE on this data
                                    from sklearn.metrics import r2_score, mean_squared_error
                                    r2 = r2_score(actual, predictions)
                                    rmse = np.sqrt(mean_squared_error(actual, predictions))
                                    
                                    col1, col2 = st.columns(2)
                                    col1.metric("R² on this data", round(r2, 4))
                                    col2.metric("RMSE", round(rmse, 4))
                                    
                                    # Scatter plot
                                    fig = px.scatter(
                                        x=actual, y=predictions,
                                        labels={'x': f'Actual {target}', 'y': f'Predicted {target}'},
                                        title="Actual vs Predicted",
                                        color_discrete_sequence=['#00A896'],
                                        opacity=0.6
                                    )
                                    min_v = min(actual.min(), predictions.min())
                                    max_v = max(actual.max(), predictions.max())
                                    fig.add_trace(go.Scatter(
                                        x=[min_v, max_v], y=[min_v, max_v],
                                        mode='lines', line=dict(color='red', dash='dash'),
                                        name='Perfect'
                                    ))
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                else:  # Classification
                                    from sklearn.metrics import accuracy_score
                                    acc = accuracy_score(actual, predictions)
                                    st.metric("Accuracy on this data", f"{round(acc * 100, 2)}%")
                            
                            # Show and download
                            st.markdown("### Full results")
                            st.dataframe(result_df.head(100), use_container_width=True)
                            
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download all predictions as CSV",
                                data=csv,
                                file_name=f"dataset_predictions_{target}.csv",
                                mime="text/csv",
                                type="primary"
                            )
                        
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
 
 
 # ============================================================
    # TAB 8: Clustering
    # ============================================================
    with tab_cluster:
        from modules.ml_advanced2 import (
            run_kmeans, run_dbscan, run_hierarchical, find_optimal_k,
            apply_smote, apply_undersampling
        )
        
        st.subheader("🎯 Clustering")
        st.markdown("Unsupervised learning — find groups in your data without labels")
        
        cluster_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
        
        numeric_cols_list = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols_list:
            st.warning("No numeric columns. Preprocess your data first.")
        else:
            clust_tab1, clust_tab2, clust_tab3 = st.tabs([
                "🎯 Run Clustering",
                "🔍 Find Optimal K",
                "⚖️ Imbalanced Data"
            ])
            
            # ----- Sub-tab 1: Run Clustering -----
            with clust_tab1:
                cluster_features = st.multiselect(
                    "Select features for clustering:",
                    numeric_cols_list,
                    default=numeric_cols_list[:min(4, len(numeric_cols_list))],
                    key="cluster_features"
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    method = st.selectbox(
                        "Method:",
                        ["K-Means", "DBSCAN", "Hierarchical"],
                        key="cluster_method"
                    )
                
                with c2:
                    if method in ["K-Means", "Hierarchical"]:
                        n_clusters = st.slider("Number of clusters:", 2, 15, 3, key="n_clusters")
                    else:
                        n_clusters = None
                
                if method == "DBSCAN":
                    c3, c4 = st.columns(2)
                    eps = c3.slider("Epsilon (distance):", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
                    min_samples = c4.slider("Min samples:", 2, 20, 5, key="dbscan_min")
                
                if st.button("🚀 Run clustering", type="primary", key="run_cluster"):
                    if not cluster_features:
                        st.error("Select features")
                    else:
                        X = cluster_df[cluster_features].dropna()
                        
                        with st.spinner(f"Running {method}..."):
                            if method == "K-Means":
                                result = run_kmeans(X, n_clusters=n_clusters)
                            elif method == "DBSCAN":
                                result = run_dbscan(X, eps=eps, min_samples=min_samples)
                            else:
                                result = run_hierarchical(X, n_clusters=n_clusters)
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.session_state.cluster_result = {
                                "result": result,
                                "features": cluster_features,
                                "X": X
                            }
                            st.rerun()
                
                # Display results
                if "cluster_result" in st.session_state:
                    cr = st.session_state.cluster_result
                    result = cr["result"]
                    X = cr["X"]
                    
                    st.markdown("---")
                    st.markdown("### 📊 Clustering Results")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Method", result["method"])
                    c2.metric("Clusters found", result["n_clusters"])
                    if result.get("silhouette") is not None:
                        c3.metric("Silhouette Score", result["silhouette"],
                                  help="Higher is better (max 1.0)")
                    
                    if "n_noise" in result:
                        st.info(f"DBSCAN found {result['n_noise']} noise points (outliers)")
                    
                    # Visualize clusters (2D scatter using first 2 features)
                    if len(cr["features"]) >= 2:
                        plot_df = X.copy().reset_index(drop=True)
                        plot_df["Cluster"] = result["labels"].astype(str)
                        
                        fig = px.scatter(
                            plot_df,
                            x=cr["features"][0],
                            y=cr["features"][1],
                            color="Cluster",
                            title=f"{result['method']} Clusters (2D view)",
                            opacity=0.7
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 3D view if 3+ features
                        if len(cr["features"]) >= 3:
                            fig_3d = px.scatter_3d(
                                plot_df,
                                x=cr["features"][0],
                                y=cr["features"][1],
                                z=cr["features"][2],
                                color="Cluster",
                                title=f"{result['method']} Clusters (3D view)",
                                opacity=0.7
                            )
                            fig_3d.update_layout(height=600)
                            st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Cluster sizes
                    st.markdown("#### Cluster sizes")
                    cluster_counts = pd.Series(result["labels"]).value_counts().sort_index().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)
            
            # ----- Sub-tab 2: Find Optimal K -----
            with clust_tab2:
                st.markdown("### Find the optimal number of clusters")
                st.markdown("Uses the Elbow Method and Silhouette Score")
                
                optimal_features = st.multiselect(
                    "Features:",
                    numeric_cols_list,
                    default=numeric_cols_list[:min(4, len(numeric_cols_list))],
                    key="optimal_features"
                )
                max_k = st.slider("Max K to test:", 3, 15, 10, key="max_k")
                
                if st.button("🔍 Find optimal K", type="primary", key="find_k"):
                    if not optimal_features:
                        st.error("Select features")
                    else:
                        X = cluster_df[optimal_features].dropna()
                        
                        with st.spinner("Testing different K values..."):
                            k_results = find_optimal_k(X, max_k=max_k)
                        
                        if "error" in k_results.columns:
                            st.error(k_results["error"].iloc[0])
                        else:
                            st.dataframe(k_results, use_container_width=True, hide_index=True)
                            
                            c1, c2 = st.columns(2)
                            
                            with c1:
                                fig_elbow = px.line(
                                    k_results, x="K", y="Inertia",
                                    title="Elbow Method",
                                    markers=True,
                                    color_discrete_sequence=['#00A896']
                                )
                                fig_elbow.update_layout(height=400)
                                st.plotly_chart(fig_elbow, use_container_width=True)
                            
                            with c2:
                                fig_sil = px.line(
                                    k_results, x="K", y="Silhouette",
                                    title="Silhouette Score",
                                    markers=True,
                                    color_discrete_sequence=['#1E2761']
                                )
                                fig_sil.update_layout(height=400)
                                st.plotly_chart(fig_sil, use_container_width=True)
                            
                            # Recommendation
                            best_k = k_results.loc[k_results["Silhouette"].idxmax(), "K"]
                            st.success(f"🎯 **Recommended K = {best_k}** (highest silhouette score)")
            
            # ----- Sub-tab 3: Imbalanced Data -----
            with clust_tab3:
                st.markdown("### Handle imbalanced classification data")
                st.markdown("SMOTE creates synthetic samples for minority classes")
                
                target_imb = st.selectbox(
                    "Target variable:",
                    cluster_df.columns.tolist(),
                    key="imb_target"
                )
                
                if target_imb:
                    class_counts = cluster_df[target_imb].value_counts()
                    st.markdown("#### Current class distribution")
                    
                    fig = px.bar(
                        x=class_counts.index.astype(str),
                        y=class_counts.values,
                        title="Class distribution",
                        labels={"x": "Class", "y": "Count"},
                        color=class_counts.values,
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    imbalance_ratio = class_counts.max() / class_counts.min()
                    if imbalance_ratio > 2:
                        st.warning(f"⚠️ Imbalance ratio: {round(imbalance_ratio, 2)}:1 — consider balancing")
                    else:
                        st.success(f"✅ Classes are relatively balanced (ratio: {round(imbalance_ratio, 2)}:1)")
                
                features_imb = st.multiselect(
                    "Features for resampling:",
                    [c for c in numeric_cols_list if c != target_imb],
                    default=[c for c in numeric_cols_list if c != target_imb][:5],
                    key="imb_features"
                )
                
                resample_method = st.radio(
                    "Method:",
                    ["SMOTE (oversample minority)", "Random undersampling (reduce majority)"],
                    key="resample_method"
                )
                
                if st.button("⚖️ Balance the data", type="primary", key="balance_btn"):
                    if not features_imb:
                        st.error("Select features")
                    else:
                        valid_mask = cluster_df[features_imb + [target_imb]].notna().all(axis=1)
                        X = cluster_df.loc[valid_mask, features_imb]
                        y = cluster_df.loc[valid_mask, target_imb]
                        
                        with st.spinner("Resampling..."):
                            if "SMOTE" in resample_method:
                                result = apply_smote(X, y)
                            else:
                                result = apply_undersampling(X, y)
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            c1, c2 = st.columns(2)
                            c1.metric("Original size", result["original_size"])
                            c2.metric("New size", result["new_size"])
                            
                            # Show before/after
                            st.markdown("#### Before vs After")
                            compare_df = pd.DataFrame({
                                "Class": list(result["before_counts"].keys()),
                                "Before": list(result["before_counts"].values()),
                                "After": [result["after_counts"].get(k, 0) for k in result["before_counts"].keys()]
                            })
                            st.dataframe(compare_df, use_container_width=True, hide_index=True)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='Before', x=compare_df["Class"].astype(str), y=compare_df["Before"], marker_color='#F96167'))
                            fig.add_trace(go.Bar(name='After', x=compare_df["Class"].astype(str), y=compare_df["After"], marker_color='#00A896'))
                            fig.update_layout(barmode='group', title="Class Balance Comparison", height=400)
                            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # TAB 9: Dimensionality Reduction
    # ============================================================
    with tab_dimred:
        from modules.ml_advanced2 import run_pca, run_tsne
        
        st.subheader("📐 Dimensionality Reduction")
        st.markdown("Reduce high-dimensional data to 2D/3D for visualization")
        
        dimred_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
        numeric_cols_list = dimred_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols_list) < 3:
            st.warning("Need at least 3 numeric columns")
        else:
            dr_tab1, dr_tab2 = st.tabs(["📊 PCA", "🎯 t-SNE"])
            
            # ----- PCA -----
            with dr_tab1:
                st.markdown("### Principal Component Analysis")
                
                pca_features = st.multiselect(
                    "Features:",
                    numeric_cols_list,
                    default=numeric_cols_list,
                    key="pca_features"
                )
                
                c1, c2 = st.columns(2)
                n_components = c1.slider("Components:", 2, min(10, len(pca_features)) if pca_features else 2, 2, key="pca_n")
                
                color_options = ["None"] + dimred_df.columns.tolist()
                color_by_pca = c2.selectbox("Color by:", color_options, key="pca_color")
                
                if st.button("🚀 Run PCA", type="primary", key="run_pca"):
                    if len(pca_features) < n_components:
                        st.error("Not enough features")
                    else:
                        X = dimred_df[pca_features].dropna()
                        
                        with st.spinner("Running PCA..."):
                            result = run_pca(X, n_components=n_components)
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Explained variance
                            st.markdown("#### Explained variance")
                            var_df = pd.DataFrame({
                                "Component": [f"PC{i+1}" for i in range(n_components)],
                                "Variance Ratio": result["explained_variance"],
                                "Cumulative": result["cumulative_variance"]
                            })
                            st.dataframe(var_df, use_container_width=True, hide_index=True)
                            
                            fig_var = px.bar(
                                var_df, x="Component", y="Variance Ratio",
                                title="Variance explained by each component",
                                color="Variance Ratio",
                                color_continuous_scale="Teal",
                                text_auto='.2%'
                            )
                            fig_var.update_layout(height=400)
                            st.plotly_chart(fig_var, use_container_width=True)
                            
                            # 2D scatter plot
                            X_reduced = result["X_reduced"]
                            plot_df = pd.DataFrame(X_reduced[:, :2], columns=["PC1", "PC2"], index=X.index)
                            
                            if color_by_pca != "None":
                                plot_df["color"] = dimred_df.loc[X.index, color_by_pca]
                                fig = px.scatter(
                                    plot_df, x="PC1", y="PC2",
                                    color="color",
                                    title="PCA 2D Projection",
                                    opacity=0.6
                                )
                            else:
                                fig = px.scatter(
                                    plot_df, x="PC1", y="PC2",
                                    title="PCA 2D Projection",
                                    opacity=0.6,
                                    color_discrete_sequence=['#00A896']
                                )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 3D if n_components >= 3
                            if n_components >= 3:
                                plot_df_3d = pd.DataFrame(X_reduced[:, :3], columns=["PC1", "PC2", "PC3"], index=X.index)
                                if color_by_pca != "None":
                                    plot_df_3d["color"] = dimred_df.loc[X.index, color_by_pca]
                                    fig_3d = px.scatter_3d(plot_df_3d, x="PC1", y="PC2", z="PC3",
                                                           color="color", title="PCA 3D")
                                else:
                                    fig_3d = px.scatter_3d(plot_df_3d, x="PC1", y="PC2", z="PC3",
                                                           title="PCA 3D", color_discrete_sequence=['#00A896'])
                                fig_3d.update_layout(height=600)
                                st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Component loadings
                            st.markdown("#### Component loadings")
                            st.caption("How much each original feature contributes to each component")
                            st.dataframe(result["loadings"].round(3), use_container_width=True)
            
            # ----- t-SNE -----
            with dr_tab2:
                st.markdown("### t-SNE Visualization")
                st.caption("Good for finding clusters but slower than PCA")
                
                tsne_features = st.multiselect(
                    "Features:",
                    numeric_cols_list,
                    default=numeric_cols_list,
                    key="tsne_features"
                )
                
                c1, c2 = st.columns(2)
                perplexity = c1.slider("Perplexity:", 5, 50, 30, key="tsne_perp")
                color_by_tsne = c2.selectbox("Color by:", ["None"] + dimred_df.columns.tolist(), key="tsne_color")
                
                if st.button("🚀 Run t-SNE", type="primary", key="run_tsne"):
                    X = dimred_df[tsne_features].dropna()
                    
                    with st.spinner("Running t-SNE... (this can be slow)"):
                        result = run_tsne(X, perplexity=perplexity)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        if result.get("sampled_count", len(X)) < len(X):
                            st.caption(f"Sampled {result['sampled_count']} of {len(X)} rows")
                        
                        plot_df = pd.DataFrame(result["X_reduced"], columns=["t-SNE 1", "t-SNE 2"])
                        plot_df.index = result.get("sampled_idx", X.index)
                        
                        if color_by_tsne != "None":
                            plot_df["color"] = dimred_df.loc[plot_df.index, color_by_tsne].values
                            fig = px.scatter(plot_df, x="t-SNE 1", y="t-SNE 2",
                                             color="color", title="t-SNE Projection", opacity=0.6)
                        else:
                            fig = px.scatter(plot_df, x="t-SNE 1", y="t-SNE 2",
                                             title="t-SNE Projection", opacity=0.6,
                                             color_discrete_sequence=['#00A896'])
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # TAB 10: Time Series Forecasting
    # ============================================================
    with tab_forecast:
        from modules.ml_advanced2 import run_prophet_forecast, run_arima_forecast
        
        st.subheader("📅 Time Series Forecasting")
        st.markdown("Predict future values using historical data")
        
        forecast_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
        
        all_cols = forecast_df.columns.tolist()
        numeric_cols_list = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox("Date column:", all_cols, key="fc_date")
        with c2:
            value_col = st.selectbox("Value column (what to forecast):", numeric_cols_list, key="fc_value")
        
        c3, c4 = st.columns(2)
        periods = c3.slider("Periods to forecast:", 7, 365, 30, key="fc_periods")
        method = c4.selectbox("Method:", ["Prophet", "ARIMA"], key="fc_method")
        
        st.info(
            "💡 The date column should contain datetime values. "
            "Prophet is great for seasonal data. ARIMA is simpler and faster."
        )
        
        if st.button("📅 Generate forecast", type="primary", key="run_forecast"):
            with st.spinner(f"Running {method}..."):
                if method == "Prophet":
                    result = run_prophet_forecast(forecast_df, date_col, value_col, periods)
                else:
                    result = run_arima_forecast(forecast_df, date_col, value_col, periods)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✅ Generated {periods}-period forecast using {method}")
                
                if method == "Prophet":
                    historical = result["historical_df"]
                    forecast = result["forecast_df"]
                    future = result["future_df"]
                    
                    # Main plot
                    fig = go.Figure()
                    
                    # Historical
                    fig.add_trace(go.Scatter(
                        x=historical['ds'], y=historical['y'],
                        mode='markers', name='Historical',
                        marker=dict(color='#1E2761', size=5)
                    ))
                    
                    # Full forecast (trend)
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', name='Forecast',
                        line=dict(color='#00A896', width=2)
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0, 168, 150, 0.2)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Confidence'
                    ))
                    
                    fig.update_layout(
                        title=f"Forecast: {value_col}",
                        xaxis_title="Date",
                        yaxis_title=value_col,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show future values table
                    st.markdown("#### Future predictions")
                    display_future = future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    display_future.columns = ['Date', 'Forecast', 'Lower', 'Upper']
                    st.dataframe(display_future.head(30), use_container_width=True, hide_index=True)
                
                else:  # ARIMA
                    historical = result["historical_df"]
                    forecast = result["forecast_df"]
                    
                    c1, c2 = st.columns(2)
                    c1.metric("AIC", result["aic"])
                    c2.metric("BIC", result["bic"])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical[result["date_col"]], y=historical[result["value_col"]],
                        mode='lines+markers', name='Historical',
                        line=dict(color='#1E2761')
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['date'], y=forecast['forecast'],
                        mode='lines+markers', name='Forecast',
                        line=dict(color='#00A896', dash='dash')
                    ))
                    fig.update_layout(title=f"ARIMA Forecast: {value_col}", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### Future predictions")
                    st.dataframe(forecast, use_container_width=True, hide_index=True)
                    
                    
                    
 # ============================================================
    # TAB 11: Project Management
    # ============================================================
    with tab_project:
        from modules.project_manager import (
            save_project, load_project, export_as_notebook, augment_numeric_data
        )
        
        st.subheader("💼 Project Management")
        st.markdown("Save your work, export notebooks, and augment data")
        
        proj_tab1, proj_tab2, proj_tab3 = st.tabs([
            "💾 Save / Load Project",
            "📓 Export Notebook",
            "🎲 Data Augmentation"
        ])
        
        # ----- Sub-tab 1: Save/Load -----
        with proj_tab1:
            st.markdown("### Save your entire project")
            st.markdown("Includes: original data, processed data, trained models, training history")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### 💾 Save")
                
                project_name = st.text_input("Project name:", value=f"datalab_project_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
                
                if st.button("💾 Download project file", type="primary", use_container_width=True):
                    with st.spinner("Saving project..."):
                        project_bytes = save_project(
                            df=st.session_state.df,
                            df_processed=st.session_state.get("df_processed"),
                            last_training=st.session_state.get("last_training"),
                            train_history=st.session_state.get("train_history", [])
                        )
                    
                    st.download_button(
                        label="📥 Click to download",
                        data=project_bytes,
                        file_name=f"{project_name}.datalab",
                        mime="application/octet-stream",
                        type="primary"
                    )
                    st.success("Project saved! Click the download button above.")
            
            with c2:
                st.markdown("#### 📂 Load")
                
                project_file = st.file_uploader(
                    "Upload .datalab file:",
                    type=["datalab", "pkl"],
                    key="project_upload"
                )
                
                if project_file is not None:
                    if st.button("📂 Load project", type="primary", use_container_width=True):
                        with st.spinner("Loading project..."):
                            result = load_project(project_file.read())
                        
                        if result["success"]:
                            project = result["project"]
                            st.session_state.df = project["df"]
                            st.session_state.df_processed = project.get("df_processed")
                            st.session_state.last_training = project.get("last_training")
                            st.session_state.train_history = project.get("train_history", [])
                            st.session_state.df_name = f"Loaded from {project_file.name}"
                            
                            st.success(f"✅ Project loaded! Saved at: {project.get('saved_at', 'unknown')}")
                            st.rerun()
                        else:
                            st.error(f"Failed to load: {result['error']}")
            
            st.markdown("---")
            
            # Current project info
            st.markdown("#### 📊 Current project state")
            info_cols = st.columns(4)
            info_cols[0].metric("Dataset", st.session_state.get("df_name", "?"))
            info_cols[1].metric("Rows", f"{len(st.session_state.df):,}" if st.session_state.df is not None else "0")
            info_cols[2].metric("Models trained", len(st.session_state.get("train_history", [])))
            info_cols[3].metric("Has processed data", "Yes" if st.session_state.get("df_processed") is not None else "No")
        
        # ----- Sub-tab 2: Export Notebook -----
        with proj_tab2:
            st.markdown("### Export as Jupyter notebook")
            st.markdown("Generate a reproducible `.ipynb` file of your workflow")
            
            if "last_training" not in st.session_state:
                st.warning("⚠️ Train a model first in ML Training tab")
            else:
                training = st.session_state.last_training
                
                st.info(
                    f"**Will export:**\n\n"
                    f"- Dataset: `{st.session_state.get('df_name', 'dataset')}`\n"
                    f"- Target: `{training['target']}`\n"
                    f"- Features: {len(training['features'])}\n"
                    f"- Model: {training['model_name']}\n"
                    f"- Problem type: {training['problem_type']}"
                )
                
                if st.button("📓 Generate notebook", type="primary", key="gen_notebook"):
                    dataset_name = st.session_state.get("df_name", "dataset").replace(" ", "_").replace("(", "").replace(")", "")
                    
                    notebook_json = export_as_notebook(
                        dataset_name=dataset_name,
                        preprocessing_steps=["Fill missing values", "Encode categorical columns"],
                        feature_engineering_steps=[],
                        target=training["target"],
                        features=training["features"],
                        model_name=training["model_name"],
                        problem_type=training["problem_type"]
                    )
                    
                    st.success("✅ Notebook generated!")
                    
                    st.download_button(
                        label="📥 Download .ipynb",
                        data=notebook_json,
                        file_name=f"{dataset_name}_workflow.ipynb",
                        mime="application/x-ipynb+json",
                        type="primary"
                    )
                    
                    with st.expander("Preview notebook content"):
                        st.code(notebook_json[:2000] + "...", language="json")
        
        # ----- Sub-tab 3: Data Augmentation -----
        with proj_tab3:
            st.markdown("### Data augmentation")
            st.markdown("Generate synthetic samples to expand small datasets")
            
            aug_df = st.session_state.get("df_processed", df) if st.session_state.get("df_processed") is not None else df
            numeric_cols_list = aug_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols_list:
                st.warning("No numeric columns")
            else:
                aug_features = st.multiselect(
                    "Columns to augment:",
                    numeric_cols_list,
                    default=numeric_cols_list[:min(5, len(numeric_cols_list))],
                    key="aug_features"
                )
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    aug_method = st.selectbox(
                        "Method:",
                        ["noise", "interpolation", "bootstrap"],
                        key="aug_method",
                        help="noise: add Gaussian noise | interpolation: blend samples | bootstrap: resample"
                    )
                with c2:
                    n_samples = st.slider("Synthetic samples:", 10, 5000, 100, 10, key="aug_n")
                with c3:
                    if aug_method == "noise":
                        noise_level = st.slider("Noise level:", 0.01, 0.3, 0.05, 0.01, key="aug_noise")
                    else:
                        noise_level = 0.05
                
                if st.button("🎲 Generate synthetic data", type="primary", key="run_aug"):
                    if not aug_features:
                        st.error("Select features")
                    else:
                        with st.spinner("Generating synthetic samples..."):
                            result = augment_numeric_data(
                                aug_df, aug_features, aug_method, n_samples, noise_level
                            )
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Original size", result["original_size"])
                            c2.metric("Synthetic samples", result["n_synthetic"])
                            c3.metric("New total", result["new_size"])
                            
                            # Compare distributions
                            st.markdown("#### Distribution comparison")
                            
                            feat_to_view = st.selectbox("View feature:", aug_features, key="aug_view")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=aug_df[feat_to_view].dropna(),
                                name='Original',
                                opacity=0.6,
                                marker_color='#1E2761'
                            ))
                            fig.add_trace(go.Histogram(
                                x=result["synthetic_df"][feat_to_view],
                                name='Synthetic',
                                opacity=0.6,
                                marker_color='#00A896'
                            ))
                            fig.update_layout(
                                barmode='overlay',
                                title=f"Original vs Synthetic: {feat_to_view}",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Preview synthetic data
                            st.markdown("#### Synthetic samples preview")
                            st.dataframe(result["synthetic_df"].head(20), use_container_width=True)
                            
                            # Option to use augmented data
                            if st.button("✅ Use augmented dataset as working data"):
                                st.session_state.df_processed = result["combined_df"]
                                st.success("Working dataset now includes synthetic samples!")
                                st.rerun()                  
# ============================================================
# Footer
# ============================================================


st.markdown("---")
st.markdown(
    "**DataLab AI** v1.0 | "
    "Built with Streamlit, pandas, scikit-learn, and Plotly"
)

