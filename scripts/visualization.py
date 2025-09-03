
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the figures folder exists
figures_folder = '../figures'
os.makedirs(figures_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv('../data/dataR2.csv')

# Fix column names if needed
df.columns = [c.replace('-', '.') for c in df.columns]

# Map classification to labels
df['Classification_label'] = df['Classification'].map({1: 'Healthy', 2: 'Breast Cancer'})

# List of numeric features
numeric_features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']



# ----------------------------- Data Exploration ----------------------------- #
# 1. Pairplot for features colored by classification
sns.pairplot(df, vars=['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1'],
             hue='Classification_label', palette='Set2')
plt.suptitle('Pairplot of features by Classification', y=1.02)
plt.savefig(os.path.join(figures_folder, 'pairplot_features.png'))
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(10,8))
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig(os.path.join(figures_folder, 'correlation_heatmap.png'))
plt.close()

# 3. Boxplots for regression features vs target (HOMA)
regression_features = ['Insulin', 'Leptin', 'Resistin', 'MCP.1']
for feature in regression_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Classification', y=feature, data=df)
    plt.title(f'{feature} distribution by Classification')
    plt.xlabel('Classification')
    plt.ylabel(feature)
    plt.savefig(os.path.join(figures_folder, f'boxplot_{feature}.png'))
    plt.close()

# ----------------------------- Regression Visualizations ----------------------------- #
# 4. Scatter plots of regression features vs target
target = 'HOMA'
for feature in regression_features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df[target], hue=df['Classification'], alpha=0.7)
    plt.title(f'{feature} vs {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.savefig(os.path.join(figures_folder, f'scatter_{feature}_vs_{target}.png'))
    plt.close()

# 5. Histogram of the target variable
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='HOMA', hue='Classification_label', kde=True, palette='Set2', alpha=0.6)
plt.title('Distribution of HOMA by Classification')
plt.xlabel('HOMA')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_folder, f'histogram_HOMA.png'))
plt.close()

# ----------------------------- Classification Visualizations ----------------------------- #
# 6. Count plot of classes
plt.figure(figsize=(6,4))
sns.countplot(x='Classification', data=df)
plt.title('Class Distribution')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_folder, 'class_distribution.png'))
plt.close()

# 7. Boxplots for each numeric feature by classification
numeric_features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
for feature in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Classification_label', y=feature, data=df, palette='Set2')
    plt.title(f'{feature} by Classification')
    plt.xlabel('Classification')
    plt.ylabel(feature)
    plt.savefig(os.path.join(figures_folder, 'boxplot_{feature}.png'))
    plt.close()

print("All visualizations have been saved in the 'figures' folder.")
