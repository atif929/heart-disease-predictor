# Heart Disease Prediction - EDA & Preprocessing
# Save this as: notebooks/eda_preprocessing.ipynb OR notebooks/eda_preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os  # Added to handle paths correctly

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# 0. SETUP PATHS (CRITICAL FIX)
# ========================================
# Get the folder where this script is currently located (e.g., .../notebooks)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root (e.g., .../heart-disease-prediction)
project_root = os.path.dirname(script_dir)

# Define exact paths to your data and images folders
data_path = os.path.join(project_root, 'data', 'heart.csv')
images_dir = os.path.join(project_root, 'images')

# Create the images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

# ========================================
# 1. LOAD DATASET
# ========================================
print("="*50)
print("LOADING DATASET")
print("="*50)

# Use the dynamic path variable we created
print(f"Reading data from: {data_path}")
df = pd.read_csv(data_path)

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Samples: {df.shape[0]}")
print(f"Number of Features: {df.shape[1]}")

# ========================================
# 2. BASIC INFORMATION
# ========================================
print("\n" + "="*50)
print("BASIC INFORMATION")
print("="*50)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

# ========================================
# 3. CHECK FOR MISSING VALUES
# ========================================
print("\n" + "="*50)
print("MISSING VALUES CHECK")
print("="*50)

missing_values = df.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values)
print(f"\nTotal Missing Values: {missing_values.sum()}")

# ========================================
# 4. CHECK FOR DUPLICATES
# ========================================
print("\n" + "="*50)
print("DUPLICATE CHECK")
print("="*50)

duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# ========================================
# 5. TARGET VARIABLE ANALYSIS
# ========================================
print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)

print("\nTarget Distribution:")
print(df['target'].value_counts())
print("\nTarget Percentage:")
print(df['target'].value_counts(normalize=True) * 100)

# Plot target distribution
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Distribution of Heart Disease', fontsize=16, fontweight='bold')
plt.xlabel('Target (0: No Disease, 1: Disease)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()

# Save using the dynamic path
save_path = os.path.join(images_dir, 'target_distribution.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")
# plt.show() # Commented out to prevent blocking the script if running automatically

# ========================================
# 6. FEATURE DISTRIBUTIONS
# ========================================
print("\n" + "="*50)
print("FEATURE DISTRIBUTIONS")
print("="*50)

# Numerical features
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black')
    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
save_path = os.path.join(images_dir, 'feature_distributions.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# ========================================
# 7. CORRELATION ANALYSIS
# ========================================
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

# Correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation with Target:")
print(correlation_matrix['target'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Matrix of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(images_dir, 'correlation_heatmap.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# ========================================
# 8. FEATURE vs TARGET ANALYSIS
# ========================================
print("\n" + "="*50)
print("FEATURE vs TARGET ANALYSIS")
print("="*50)

# Age vs Target
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='age', data=df, palette='Set2')
plt.title('Age Distribution by Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Target (0: No Disease, 1: Disease)')
plt.ylabel('Age')
plt.tight_layout()
save_path = os.path.join(images_dir, 'age_vs_target.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# Chest Pain Type vs Target
plt.figure(figsize=(10, 6))
pd.crosstab(df['cp'], df['target']).plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Chest Pain Type vs Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)
plt.tight_layout()
save_path = os.path.join(images_dir, 'cp_vs_target.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# Sex vs Target
plt.figure(figsize=(10, 6))
pd.crosstab(df['sex'], df['target']).plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Gender vs Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)
plt.tight_layout()
save_path = os.path.join(images_dir, 'sex_vs_target.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# ========================================
# 9. OUTLIER DETECTION
# ========================================
print("\n" + "="*50)
print("OUTLIER DETECTION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'Boxplot of {col}', fontweight='bold')
    axes[idx].set_ylabel(col)

plt.tight_layout()
save_path = os.path.join(images_dir, 'outlier_detection.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

# Print outlier statistics
for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"\n{col}: {len(outliers)} outliers detected")

# ========================================
# 10. PAIRPLOT (KEY FEATURES)
# ========================================
print("\n" + "="*50)
print("CREATING PAIRPLOT")
print("="*50)

key_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
sns.pairplot(df[key_features], hue='target', palette='Set1', diag_kind='kde')
save_path = os.path.join(images_dir, 'pairplot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {save_path}")

print("\n" + "="*50)
print(f"EDA COMPLETE! All visualizations saved to: {images_dir}")
print("="*50)