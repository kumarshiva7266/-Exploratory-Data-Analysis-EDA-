import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Load sample dataset (Iris dataset)
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# 1. Basic Data Overview
print("\n=== Basic Data Overview ===")
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 2. Univariate Analysis
print("\n=== Univariate Analysis ===")

# Create a figure with subplots for histograms
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=column, hue='species', kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# Create boxplots
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='species', y=column)
    plt.title(f'Boxplot of {column} by Species')
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()

# 3. Bivariate Analysis
print("\n=== Bivariate Analysis ===")

# Create pairplot
sns.pairplot(df, hue='species')
plt.savefig('pairplot.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# 4. Interactive Visualizations using Plotly
print("\n=== Interactive Visualizations ===")

# Create interactive scatter plot
fig = px.scatter(df, x='sepal_length', y='sepal_width', 
                 color='species', title='Sepal Length vs Width')
fig.write_html('scatter_plot.html')

# Create interactive box plots
fig = go.Figure()
for species in df['species'].unique():
    fig.add_trace(go.Box(
        y=df[df['species'] == species]['sepal_length'],
        name=species
    ))
fig.update_layout(title='Interactive Box Plot of Sepal Length by Species')
fig.write_html('interactive_boxplot.html')

# 5. Advanced Analysis
print("\n=== Advanced Analysis ===")

# Calculate additional statistics
print("\nSkewness of numeric features:")
print(df.iloc[:, :-1].skew())

print("\nKurtosis of numeric features:")
print(df.iloc[:, :-1].kurtosis())

# Create violin plots
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.violinplot(data=df, x='species', y=column)
    plt.title(f'Violin Plot of {column} by Species')
plt.tight_layout()
plt.savefig('violin_plots.png')
plt.close()

print("\nEDA Analysis Complete! Check the generated visualizations:")
print("1. histograms.png - Distribution of features")
print("2. boxplots.png - Box plots of features by species")
print("3. pairplot.png - Pairwise relationships between features")
print("4. correlation_matrix.png - Correlation heatmap")
print("5. scatter_plot.html - Interactive scatter plot")
print("6. interactive_boxplot.html - Interactive box plot")
print("7. violin_plots.png - Violin plots showing distribution shapes") 