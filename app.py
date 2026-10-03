import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from sklearn.linear_model import LinearRegression

# ==========================================
# PAGE CONFIGURATION & CUSTOM GLASSMORPHISM CSS
# ==========================================
st.set_page_config(
    page_title="Weather Data Analysis Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    :root {
        --bg: #f3f7f1;
        --panel: #ffffff;
        --panel-alt: #eef7ee;
        --soft-green: #dfeee0;
        --forest: #2e7d5d;
        --deep: #234f3c;
        --sage: #8bbf9f;
        --teal: #31a39a;
        --amber: #d6a64d;
        --muted: #5a6d67;
        --text: #18332a;
        --border: rgba(28, 58, 49, 0.12);
        --shadow: rgba(38, 74, 59, 0.12);
    }

    .stApp {
        background: linear-gradient(180deg, #f7faf6 0%, #edf5ef 100%);
        color: var(--text);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .main-title {
        color: var(--deep);
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.04em;
    }

    .sub-title {
        color: var(--muted);
        font-size: 1.08rem;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(238,247,238,0.96));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem 1.1rem;
        box-shadow: 0 12px 28px var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(46, 125, 93, 0.25);
        box-shadow: 0 16px 32px rgba(46, 125, 93, 0.12);
    }

    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--forest);
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbf8 0%, #edf5ed 100%) !important;
        border-right: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.66);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre;
        border-radius: 10px;
        color: var(--muted);
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #9ad5b6 0%, #5abf9a 100%) !important;
        color: var(--deep) !important;
        box-shadow: 0 8px 18px rgba(90, 191, 154, 0.2);
    }

    .info-box {
        background: linear-gradient(90deg, rgba(154, 213, 182, 0.16), rgba(49, 163, 154, 0.08));
        border-left: 4px solid var(--forest);
        padding: 12px 16px;
        border-radius: 12px;
        color: var(--deep);
        margin-bottom: 1rem;
    }

    .stMultiSelect [data-baseweb="tag"],
    .stMultiSelect [data-baseweb="pill"],
    div[data-baseweb="select"] span[data-baseweb="tag"],
    div[data-baseweb="select"] div[data-baseweb="pill"] {
        background: linear-gradient(135deg, rgba(180, 221, 194, 0.92), rgba(110, 188, 146, 0.9)) !important;
        border: 1px solid rgba(46, 125, 93, 0.22) !important;
        color: var(--deep) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .stMultiSelect [data-baseweb="tag"] *,
    .stMultiSelect [data-baseweb="pill"] *,
    div[data-baseweb="select"] span[data-baseweb="tag"] *,
    div[data-baseweb="select"] div[data-baseweb="pill"] * {
        color: var(--deep) !important;
    }

    .stSlider [data-testid="stBaseSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #9ad5b6 0%, #5abf9a 100%) !important;
    }

    .stSlider [data-testid="stBaseSlider"] [data-testid="stThumbValue"] {
        background: #2e7d5d !important;
        border: 2px solid #ffffff !important;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div.stFormSubmitButton > button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #9ad5b6 0%, #5abf9a 100%) !important;
        color: var(--deep) !important;
        border: 1px solid rgba(46, 125, 93, 0.25) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 18px rgba(90, 191, 154, 0.18) !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    div.stFormSubmitButton > button:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7ec9a0 0%, #4caf8d 100%) !important;
        border-color: rgba(46, 125, 93, 0.35) !important;
        color: var(--deep) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS & DATA LOADING
# ==========================================
@st.cache_data
def clean_column_name(column_name):
    text = str(column_name).strip()
    text = text.replace("langlitude", "latitude").replace("lattitude", "latitude")
    text = text.replace("longlattitude", "longitude").replace("longitud", "longitude")
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return text.title()


@st.cache_data
def drop_geo_columns(df):
    if df is None:
        return df
    geo_aliases = (
        "latitude", "longitude", "lat", "lon",
        "langlitude", "lattitude", "longlattitude", "longitud"
    )
    cols_to_drop = [
        col for col in df.columns
        if any(alias in str(col).lower() for alias in geo_aliases)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


@st.cache_data
def load_data(uploaded_file, default_path="sample_data.csv"):
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return None
            df.columns = [clean_column_name(col) for col in df.columns]
            return drop_geo_columns(df)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
        df.columns = [clean_column_name(col) for col in df.columns]
        return drop_geo_columns(df)
    else:
        # Fallback inline demo dataset
        data = {
            'year': list(range(2000, 2024)) * 3,
            'country': ['India']*24 + ['USA']*24 + ['Brazil']*24,
            'region': ['South Asia']*24 + ['North America']*24 + ['Latin America']*24,
            'average_temperature': np.random.normal(27, 3, 72).round(1),
            'co2_emissions': np.random.normal(2.5, 0.8, 72).round(2),
            'rainfall': np.random.normal(1200, 200, 72).round(0),
            'humidity': np.random.randint(55, 95, 72),
            'sea_level': np.random.normal(0.3, 0.08, 72).round(2),
            'climate_category': np.random.choice(['Hot', 'Temperate', 'Tropical'], 72)
        }
        df = pd.DataFrame(data)
        df.columns = [clean_column_name(col) for col in df.columns]
        return drop_geo_columns(df)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8;font-size:1.5rem;'>⚡ Data Control Center</h2>", unsafe_allow_html=True)
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Custom Dataset (CSV, XLSX, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload any weather, climate, or environmental dataset for dynamic analysis!"
    )
    
    st.divider()
    
    # Load dataset
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.subheader("🔍 Global Filters")
        df_filtered = df_raw.copy()
        
        # Categorical Filter Auto-detector
        cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols[:3]: # Limit to top 3 categorical columns for sidebar simplicity
            unique_vals = df_filtered[col].unique().tolist()
            selected_vals = st.multiselect(f"Filter by {col.title()}", unique_vals, default=unique_vals)
            df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
            
        # Year or Numeric Range Filter if present
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if 'year' in [c.lower() for c in num_cols]:
            year_col = [c for c in num_cols if c.lower() == 'year'][0]
            min_yr, max_yr = int(df_filtered[year_col].min()), int(df_filtered[year_col].max())
            if min_yr < max_yr:
                selected_years = st.slider(f"Year Range ({year_col.title()})", min_yr, max_yr, (min_yr, max_yr))
                df_filtered = df_filtered[(df_filtered[year_col] >= selected_years[0]) & (df_filtered[year_col] <= selected_years[1])]

# Handle missing or empty dataset
if df_raw is None or df_raw.empty:
    st.error("No valid dataset loaded. Please upload a CSV, Excel, or JSON dataset.")
    st.stop()

# Use filtered dataframe for analysis
df = df_filtered.copy()

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("<h1 class='main-title'>🌤️ Weather Data Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Interactive Data Analytics, Dynamic Dataset Exploration & Predictive Trend Visualization</p>", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class='info-box'>
        📌 <b>Currently viewing Demo Weather Dataset.</b> Upload your own CSV / Excel dataset using the left sidebar to dynamically analyze any weather or environmental dataset!
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# METRIC CARDS OVERVIEW
# ==========================================
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Total Columns</div>
        <div class="metric-value">{df.shape[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Numeric Fields</div>
        <div class="metric-value">{len(numeric_columns)}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Categorical Fields</div>
        <div class="metric-value">{len(categorical_columns)}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    missing_pct = round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Missing Cells %</div>
        <div class="metric-value">{missing_pct}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ==========================================
# MAIN TAB NAVIGATION
# ==========================================
tab_overview, tab_eda, tab_viz, tab_custom, tab_ml, tab_export = st.tabs([
    "📊 Overview & Preview", 
    "📁 Data Profiler & Cleaning", 
    "📈 Visual Analytics Hub", 
    "🎨 Custom Chart Builder", 
    "🔮 Trend Forecasting (ML)",
    "💾 Data Export"
])

# ------------------------------------------
# TAB 1: OVERVIEW & DATA PREVIEW
# ------------------------------------------
with tab_overview:
    st.subheader("📋 Dataset Preview & Quick Summary")
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("##### First 10 Rows")
        st.dataframe(df.head(10), width="stretch")
    
    with col_right:
        st.markdown("##### Column Data Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(dt) for dt in df.dtypes],
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(dtype_df, width="stretch", height=360)
        
    st.divider()
    st.subheader("⚡ Quick Key Statistics")
    if numeric_columns:
        st.dataframe(df[numeric_columns].describe().T.style.highlight_max(axis=0, color='rgba(56, 189, 248, 0.3)'), width="stretch")

# ------------------------------------------
# TAB 2: DATA PROFILER & CLEANING
# ------------------------------------------
with tab_eda:
    st.subheader("📁 Comprehensive Data Profiler")
    
    col_prof1, col_prof2 = st.columns(2)
    
    with col_prof1:
        st.markdown("##### Missing Values Heatmap")
        if df.isnull().sum().sum() == 0:
            st.success("🎉 Excellent! Your dataset has zero missing values.")
        else:
            fig_null = px.imshow(df.isnull(), labels=dict(x="Columns", y="Rows", color="Is Null"),
                                 color_continuous_scale=["#edf7f0", "#2e7d5d"], title="Missing Value Distribution")
            fig_null.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_null, width="stretch")
            
    with col_prof2:
        st.markdown("##### Column Skewness & Kurtosis Profile")
        if len(numeric_columns) > 0:
            skew_kurt_df = pd.DataFrame({
                "Column": numeric_columns,
                "Mean": df[numeric_columns].mean().round(2),
                "Median": df[numeric_columns].median().round(2),
                "Std Dev": df[numeric_columns].std().round(2),
                "Skewness": df[numeric_columns].skew().round(2),
                "Kurtosis": df[numeric_columns].kurtosis().round(2)
            })
            st.dataframe(skew_kurt_df, width="stretch", height=350)
            
    st.divider()
    st.subheader("🛠️ Quick Data Cleaning Tools")
    clean_col1, clean_col2 = st.columns(2)
    
    with clean_col1:
        st.markdown("##### Handle Missing Values")
        null_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        if null_cols:
            selected_null_col = st.selectbox("Select column to fix missing values:", null_cols)
            clean_method = st.radio("Choose Imputation Method:", ["Fill Mean/Mode", "Fill Median", "Forward Fill", "Drop Rows with Missing Values"], key="impute_method")
            if st.button("Apply Imputation"):
                if clean_method == "Fill Mean/Mode":
                    if selected_null_col in numeric_columns:
                        df[selected_null_col] = df[selected_null_col].fillna(df[selected_null_col].mean())
                    else:
                        df[selected_null_col] = df[selected_null_col].fillna(df[selected_null_col].mode()[0])
                elif clean_method == "Fill Median":
                    if selected_null_col in numeric_columns:
                        df[selected_null_col] = df[selected_null_col].fillna(df[selected_null_col].median())
                elif clean_method == "Forward Fill":
                    df[selected_null_col] = df[selected_null_col].ffill()
                elif clean_method == "Drop Rows with Missing Values":
                    df = df.dropna(subset=[selected_null_col])
                st.success(f"Updated missing values for '{selected_null_col}'.")
                st.rerun()
        else:
            st.info("No missing values detected in the selected subset.")

    with clean_col2:
        st.markdown("##### Remove Duplicate Rows")
        dup_count = df.duplicated().sum()
        st.write(f"Detected **{dup_count}** duplicate rows.")
        if dup_count > 0:
            if st.button("Drop Duplicate Rows"):
                df = df.drop_duplicates()
                st.success("Successfully removed duplicate rows.")
                st.rerun()

# ------------------------------------------
# TAB 3: VISUAL ANALYTICS HUB
# ------------------------------------------
with tab_viz:
    st.subheader("📈 Weather & Climate Visual Analytics")
    
    if len(numeric_columns) < 2:
        st.warning("At least 2 numerical columns are required for visualization hub.")
    else:
        # Row 1: Time Series & Multi-Variable Trend
        st.markdown("### 1️⃣ Time Series & Trend Analysis")
        ts_col1, ts_col2 = st.columns([3, 1])
        
        # Identify date/year column if available
        time_candidates = [c for c in df.columns if 'year' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
        x_time = time_candidates[0] if time_candidates else df.columns[0]
        
        with ts_col2:
            y_trend_col = st.selectbox("Select Weather Variable for Trend:", numeric_columns, index=0)
            color_group = st.selectbox("Group By (Color):", [None] + categorical_columns)
            enable_ma = st.checkbox("Show 3-Period Moving Average", value=True)
            
        with ts_col1:
            eco_palette = ["#2E7D5D", "#7CBF9E", "#4DB6AC", "#F0B24A", "#8BC34A", "#1E9D8A"]
            if color_group:
                fig_ts = px.line(df, x=x_time, y=y_trend_col, color=color_group, markers=True,
                                 title=f"{y_trend_col.replace('_', ' ').title()} over {x_time.title()} by {color_group.title()}",
                                 color_discrete_sequence=eco_palette)
            else:
                fig_ts = px.line(df, x=x_time, y=y_trend_col, markers=True,
                                 title=f"{y_trend_col.replace('_', ' ').title()} Trend over {x_time.title()}",
                                 color_discrete_sequence=["#2E7D5D"])
                if enable_ma:
                    df_sorted = df.sort_values(by=x_time)
                    ma_series = df_sorted[y_trend_col].rolling(window=3).mean()
                    fig_ts.add_scatter(x=df_sorted[x_time], y=ma_series, name="3-Period Moving Avg", line=dict(dash='dash', color='#D6A64D', width=3))
            
            fig_ts.update_layout(template="plotly_white", height=420, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                 font=dict(color="#18332a"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ts, width="stretch")
            
        st.divider()
        
        # Row 2: Correlation Matrix & Seaborn Chart Toggle
        st.markdown("### 2️⃣ Variable Correlation Matrix")
        corr_col1, corr_col2 = st.columns([1, 1])
        
        with corr_col1:
            st.markdown("##### Interactive Plotly Correlation Heatmap")
            corr_matrix = df[numeric_columns].corr().round(2)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=[[0, '#edf7ef'], [0.5, '#8fd0af'], [1, '#2e7d5d']],
                                 title="Variable Relationship Heatmap")
            fig_corr.update_layout(template="plotly_white", height=400, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                 font=dict(color="#18332a"))
            st.plotly_chart(fig_corr, width="stretch")
            
        with corr_col2:
            st.markdown("##### High-Res Seaborn Correlation Matrix")
            fig_sns, ax = plt.subplots(figsize=(8, 6))
            fig_sns.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            sns.heatmap(corr_matrix, annot=True, cmap="mako", fmt=".2f", ax=ax, cbar=True,
                        annot_kws={"size": 10, "color": "white"})
            ax.tick_params(colors='white')
            st.pyplot(fig_sns)
            plt.close(fig_sns)
            
        st.divider()
        
        # Row 3: Distribution & Boxplots
        st.markdown("### 3️⃣ Feature Distribution & Outlier Detection")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            selected_dist = st.selectbox("Select Variable for Histogram & KDE:", numeric_columns, index=0)
            fig_hist = px.histogram(df, x=selected_dist, marginal="box", color=color_group if color_group else None,
                                    title=f"Distribution of {selected_dist.title()}",
                                    color_discrete_sequence=["#2E7D5D", "#4DB6AC", "#F0B24A", "#8BBF9F"])
            fig_hist.update_layout(template="plotly_white", height=380, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                 font=dict(color="#18332a"))
            st.plotly_chart(fig_hist, width="stretch")
            
        with dist_col2:
            if categorical_columns:
                cat_var = st.selectbox("Categorical Group for Boxplot:", categorical_columns, index=0)
                selected_box = st.selectbox("Select Variable for Boxplot:", numeric_columns, index=1 if len(numeric_columns)>1 else 0)
                fig_box = px.box(df, x=cat_var, y=selected_box, color=cat_var, points="all",
                                 title=f"Boxplot of {selected_box.title()} by {cat_var.title()}",
                                 color_discrete_sequence=["#2E7D5D", "#4DB6AC", "#D6A64D", "#8BC34A"])
                fig_box.update_layout(template="plotly_white", height=380, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                 font=dict(color="#18332a"))
                st.plotly_chart(fig_box, width="stretch")
            else:
                st.info("No categorical columns available for grouping boxplots.")

        # Geographic Scatter Map if coordinates exist
        lat_cols = [c for c in df.columns if 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if 'lon' in c.lower()]
        if lat_cols and lon_cols:
            st.divider()
            st.markdown("### 4️⃣ Geographic Map Distribution")
            lat_col, lon_col = lat_cols[0], lon_cols[0]
            val_map_col = st.selectbox("Select Map Color Variable:", numeric_columns)
            fig_map = px.scatter_geo(df, lat=lat_col, lon=lon_col, color=val_map_col,
                                     hover_name=df.columns[0], size=val_map_col if (df[val_map_col] > 0).all() else None,
                                     projection="natural earth", title=f"Geographic Mapping of {val_map_col}",
                                     color_continuous_scale=[[0, '#edf7ef'], [0.5, '#9ad5b6'], [1, '#2e7d5d']])
            fig_map.update_layout(template="plotly_white", height=450, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                 font=dict(color="#18332a"))
            st.plotly_chart(fig_map, width="stretch")

# ------------------------------------------
# TAB 4: CUSTOM CHART BUILDER
# ------------------------------------------
with tab_custom:
    st.subheader("🎨 Custom Interactive Chart Generator")
    st.markdown("Build and customize any visual plot dynamically on your dataset.")
    
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
    
    with cfg_col1:
        chart_type = st.selectbox("Chart Type", [
            "Scatter Plot", "Line Plot", "Bar Chart", "Area Chart", 
            "Box Plot", "Histogram", "Pie / Donut Chart", "3D Scatter Plot"
        ])
    with cfg_col2:
        x_var = st.selectbox("X-Axis Variable", df.columns.tolist(), index=0)
    with cfg_col3:
        y_var = st.selectbox("Y-Axis Variable", numeric_columns, index=min(1, len(numeric_columns)-1))
    with cfg_col4:
        c_var = st.selectbox("Color / Grouping Variable (Optional)", [None] + df.columns.tolist())
        
    plot_title = st.text_input("Custom Chart Title:", value=f"{chart_type}: {y_var} vs {x_var}")
    
    st.divider()
    
    try:
        if chart_type == "Scatter Plot":
            fig_custom = px.scatter(df, x=x_var, y=y_var, color=c_var, trendline="ols" if len(numeric_columns)>=2 else None, title=plot_title)
        elif chart_type == "Line Plot":
            fig_custom = px.line(df, x=x_var, y=y_var, color=c_var, markers=True, title=plot_title)
        elif chart_type == "Bar Chart":
            fig_custom = px.bar(df, x=x_var, y=y_var, color=c_var, barmode="group", title=plot_title)
        elif chart_type == "Area Chart":
            fig_custom = px.area(df, x=x_var, y=y_var, color=c_var, title=plot_title)
        elif chart_type == "Box Plot":
            fig_custom = px.box(df, x=x_var, y=y_var, color=c_var, title=plot_title)
        elif chart_type == "Histogram":
            fig_custom = px.histogram(df, x=x_var, color=c_var, title=plot_title)
        elif chart_type == "Pie / Donut Chart":
            fig_custom = px.pie(df, names=x_var, values=y_var, hole=0.4, title=plot_title)
        elif chart_type == "3D Scatter Plot":
            z_var = st.selectbox("Z-Axis Variable for 3D:", numeric_columns, index=min(2, len(numeric_columns)-1))
            fig_custom = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, color=c_var, title=plot_title)
            
        fig_custom.update_layout(template="plotly_white", height=550, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                font=dict(color="#18332a"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_custom, width="stretch")
    except Exception as e:
        st.error(f"Could not render chart with selected parameters: {e}")

# ------------------------------------------
# TAB 5: TREND FORECASTING (MACHINE LEARNING)
# ------------------------------------------
with tab_ml:
    st.subheader("🔮 Machine Learning Trend Forecasting & Weather Risk Analytics")
    st.markdown("Linear Trend Modeling & Hazard Operational Indicators for Future Projection.")
    
    if len(numeric_columns) >= 2:
        ml_col1, ml_col2 = st.columns([1, 2])
        
        with ml_col1:
            st.markdown("##### ⚙️ Model Setup")
            time_cols = [c for c in df.columns if 'year' in c.lower() or 'date' in c.lower()]
            feature_x = st.selectbox("Select Independent Variable (e.g. Year/Time):", numeric_columns, index=0)
            target_y = st.selectbox("Select Target Variable to Forecast:", numeric_columns, index=min(1, len(numeric_columns)-1))
            future_steps = st.slider("Forecast Future Steps / Years Ahead:", 1, 15, 5)
            
            # Simple Linear Regression Fit
            X = df[[feature_x]].dropna()
            y = df.loc[X.index, target_y]
            
            model = LinearRegression()
            model.fit(X, y)
            r2_score = model.score(X, y)
            slope = model.coef_[0]
            
            st.markdown(f"""
            <div class="glass-card" style="margin-top:1rem;">
                <div class="metric-label">Model R² Score</div>
                <div class="metric-value">{round(r2_score, 3)}</div>
                <p style="color:#94a3b8;font-size:0.9rem;margin-top:0.5rem;">
                    Slope: <b>{round(slope, 4)}</b> per unit change
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with ml_col2:
            st.markdown("##### 📉 Trend Prediction Curve")
            # Generate future predictions
            last_val = int(df[feature_x].max())
            future_X = np.array([[last_val + i] for i in range(1, future_steps + 1)])
            future_preds = model.predict(future_X)
            
            df_hist = df[[feature_x, target_y]].sort_values(by=feature_x)
            df_future = pd.DataFrame({
                feature_x: future_X.flatten(),
                target_y: future_preds
            })
            
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=df_hist[feature_x], y=df_hist[target_y], mode='lines+markers', name='Historical Data', line=dict(color='#2E7D5D', width=3)))
            fig_ml.add_trace(go.Scatter(x=df_future[feature_x], y=df_future[target_y], mode='lines+markers', name='Linear Forecast', line=dict(color='#D6A64D', width=3, dash='dash')))
            fig_ml.update_layout(title=f"Predictive Trend for {target_y.title()} over {feature_x.title()}", template="plotly_white",
                                height=450, paper_bgcolor="#f8fbf8", plot_bgcolor="#f8fbf8",
                                font=dict(color="#18332a"))
            st.plotly_chart(fig_ml, width="stretch")
            
        st.divider()
        st.markdown("##### 🚨 Operational Weather Hazard Alert Thresholds")
        h_col1, h_col2, h_col3 = st.columns(3)
        
        # Check standard weather variables if present
        temp_cols = [c for c in numeric_columns if 'temp' in c.lower()]
        rain_cols = [c for c in numeric_columns if 'rain' in c.lower()]
        hum_cols = [c for c in numeric_columns if 'hum' in c.lower()]
        
        with h_col1:
            if temp_cols:
                max_temp = df[temp_cols[0]].max()
                status = "🔥 High Extreme Heat Risk" if max_temp > 35 else "✅ Normal Heat Range"
                st.warning(f"**Temperature Assessment ({temp_cols[0]}):** Max recorded is {max_temp}°C. Status: {status}")
            else:
                st.info("No temperature column detected.")
                
        with h_col2:
            if rain_cols:
                max_rain = df[rain_cols[0]].max()
                status = "🌧️ Heavy Rainfall Alert" if max_rain > 1500 else "🌤️ Moderate Rainfall"
                st.info(f"**Precipitation Assessment ({rain_cols[0]}):** Max recorded is {max_rain}mm. Status: {status}")
            else:
                st.info("No rainfall column detected.")
                
        with h_col3:
            if hum_cols:
                max_hum = df[hum_cols[0]].max()
                status = "💧 High Moisture Level" if max_hum > 85 else "✅ Comfortable Humidity"
                st.success(f"**Humidity Assessment ({hum_cols[0]}):** Max recorded is {max_hum}%. Status: {status}")
            else:
                st.info("No humidity column detected.")
    else:
        st.warning("Insufficient numeric variables for machine learning prediction.")

# ------------------------------------------
# TAB 6: DATA EXPORT
# ------------------------------------------
with tab_export:
    st.subheader("💾 Export Cleaned & Filtered Data")
    st.markdown("Download your processed weather dataset in CSV or Excel format.")
    
    st.dataframe(df, width="stretch", height=350)
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv_buffer,
            file_name="processed_weather_dataset.csv",
            mime="text/csv",
            width="stretch"
        )
        
    with exp_col2:
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Weather_Data')
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Dataset as Excel (.xlsx)",
                data=excel_data,
                file_name="processed_weather_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch"
            )
        except Exception as e:
            st.error(f"Excel export unavailable: {e}. Please ensure 'openpyxl' is installed.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Weather Data Analysis Dashboard • Built with Streamlit, Plotly & Pandas</p>", unsafe_allow_html=True)
