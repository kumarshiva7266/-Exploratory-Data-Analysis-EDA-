import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("""
This dashboard helps you understand your data through:
1. Summary Statistics (mean, median, std, etc.)
2. Distribution Analysis (histograms and boxplots)
3. Feature Relationships (pairplot and correlation matrix)
4. Pattern and Anomaly Detection
5. Feature-level Inferences
""")

# Sidebar for file upload
st.sidebar.header("Data Input")
use_sample = st.sidebar.checkbox("Use Sample Iris Dataset", value=True)

if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or use the sample dataset.")
        st.stop()
else:
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# 1. Summary Statistics
st.header("1. Summary Statistics")
st.markdown("""
### Key Statistics for Numeric Features
- Mean: Average value
- Median: Middle value
- Std: Standard deviation
- Min/Max: Range of values
- Quartiles: Distribution spread
""")

# Calculate and display summary statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
summary_stats = df[numeric_cols].describe()
st.dataframe(summary_stats)

# 2. Distribution Analysis
st.header("2. Distribution Analysis")
st.markdown("""
### Understanding Data Distribution
- Histograms show the frequency distribution
- Boxplots reveal outliers and quartiles
""")

# Select column for distribution analysis
selected_column = st.selectbox("Select a numeric column for distribution analysis", numeric_cols)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histogram")
    fig = px.histogram(df, x=selected_column, 
                      color='species' if 'species' in df.columns else None,
                      title=f'Distribution of {selected_column}',
                      marginal='box')  # Add box plot on the margin
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Box Plot")
    fig = px.box(df, x='species' if 'species' in df.columns else None, 
                 y=selected_column,
                 title=f'Box Plot of {selected_column}')
    st.plotly_chart(fig, use_container_width=True)

# 3. Feature Relationships
st.header("3. Feature Relationships")
st.markdown("""
### Understanding Feature Correlations
- Correlation matrix shows relationships between numeric features
- Pairplot reveals patterns between multiple features
""")

# Correlation Matrix
st.subheader("Correlation Matrix")
corr_matrix = df[numeric_cols].corr()
fig = px.imshow(corr_matrix,
                labels=dict(color="Correlation"),
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu')
st.plotly_chart(fig, use_container_width=True)

# Pairplot
st.subheader("Pairplot")
st.markdown("""
The pairplot shows relationships between all numeric features. Look for:
- Linear relationships
- Clusters or groups
- Outliers
""")
fig = px.scatter_matrix(df, dimensions=numeric_cols,
                       color='species' if 'species' in df.columns else None,
                       title="Feature Pairplot")
st.plotly_chart(fig, use_container_width=True)

# 4. Pattern and Anomaly Detection
st.header("4. Pattern and Anomaly Detection")
st.markdown("""
### Identifying Patterns and Anomalies
- Look for unusual values in boxplots
- Check for clusters in scatter plots
- Identify trends in time series (if applicable)
""")

# Scatter plot for pattern detection
st.subheader("Scatter Plot Analysis")
x_col = st.selectbox("Select X-axis", numeric_cols)
y_col = st.selectbox("Select Y-axis", numeric_cols)

fig = px.scatter(df, x=x_col, y=y_col,
                 color='species' if 'species' in df.columns else None,
                 title=f'Pattern Analysis: {x_col} vs {y_col}',
                 marginal_x='histogram',
                 marginal_y='histogram')
st.plotly_chart(fig, use_container_width=True)

# 5. Feature-level Inferences
st.header("5. Feature-level Inferences")
st.markdown("""
### Key Insights
- Skewness: Measure of distribution asymmetry
- Kurtosis: Measure of distribution "tailedness"
- Outliers: Unusual values that may need attention
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Skewness Analysis")
    skewness = df[numeric_cols].skew()
    st.dataframe(skewness.to_frame(name='Skewness'))
    st.markdown("""
    - Positive skew: Right-tailed distribution
    - Negative skew: Left-tailed distribution
    - Near zero: Symmetric distribution
    """)

with col2:
    st.subheader("Kurtosis Analysis")
    kurtosis = df[numeric_cols].kurtosis()
    st.dataframe(kurtosis.to_frame(name='Kurtosis'))
    st.markdown("""
    - High kurtosis: Heavy tails, more outliers
    - Low kurtosis: Light tails, fewer outliers
    """)

# Add download button for the dataset
st.sidebar.header("Download")
if st.sidebar.button("Download Processed Data"):
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    ) 