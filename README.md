# Weather Data Analysis Dashboard 🌤️

An interactive, high-performance web dashboard built with **Python**, **Streamlit**, **Pandas**, **NumPy**, **Plotly**, **Seaborn**, and **Matplotlib**. 

Users can upload any custom dataset (CSV, XLSX, JSON) dynamically, perform exploratory data analysis (EDA), generate interactive data visualizations, run trend forecasting models, and export cleaned data.

---

## 🚀 Features

- **Dynamic File Ingestion**: Upload any weather, climate, or custom environmental CSV, Excel (`.xlsx`), or JSON dataset with instant auto-parsing.
- **Demo Mode**: Pre-loaded with standard climate dataset if no file is uploaded.
- **Glassmorphic Dark Theme**: Sleek UI with modern metric cards, gradient headers, and interactive tabbed interface.
- **Exploratory Data Profiler**: Summary statistics, missing value heatmap, skewness & kurtosis profiling, and 1-click missing value imputation / duplicate removal.
- **Visual Analytics Hub**:
  - Time series trends with multi-variable selection & moving averages.
  - Interactive Plotly & static Seaborn correlation heatmaps.
  - Histogram & Boxplot distribution analysis for outlier detection.
  - Geographic map visualization (automatic latitude/longitude detection).
- **Custom Interactive Chart Builder**: Choose X-axis, Y-axis, Color grouping, Size, and Chart Type (Scatter, Line, Bar, Box, Histogram, Area, Pie/Donut, 3D Scatter).
- **Predictive Trend Analytics (ML)**: Linear regression trend projection model and hazard threshold alerts.
- **Data Export Hub**: Download filtered/cleaned datasets as CSV or Excel.

---

## 🛠️ Installation & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`.
