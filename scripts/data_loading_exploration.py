import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

# -------------------- LOAD DATA --------------------
# Load the Breast Cancer dataset
data_file = '../data/dataR2.csv'  # Update path if needed
df = pd.read_csv(data_file)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# -------------------- SELECT FEATURES --------------------
# For regression, predict HOMA using these features
regression_features = ['Insulin', 'Leptin', 'Resistin', 'MCP.1']
regression_target = 'HOMA'

# For classification, the target is the class column
classification_features = df.columns.drop(['Classification', 'HOMA']).tolist()
classification_target = 'Classification'

X_reg = df[regression_features].values
y_reg = df[regression_target].values.reshape(-1, 1)

X_clf = df[classification_features].values
y_clf = df[classification_target].values

# -------------------- STANDARDIZE FEATURES --------------------
from sklearn.preprocessing import StandardScaler

scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(X_reg)

scaler_clf = StandardScaler()
X_clf = scaler_clf.fit_transform(X_clf)

# -------------------- CORRELATION ANALYSIS --------------------
plt.figure(figsize=(8,6))
sns.heatmap(df[regression_features + [regression_target]].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation matrix for regression features and target")
plt.show()

print("Data loading and preprocessing completed.\n")
