# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# Load dataset
df = pd.read_csv("Disease and symptoms dataset.csv")

# Preview dataset
df.head()

print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n",df.dtypes)

# Check missing values
df.isnull().sum()
# Fill missing values with 0 (symptom absence)
df.fillna(0, inplace=True)

# Verify
df.isnull().sum()

print("Duplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates:", df.shape)

# Identify target and feature columns
target_col = df.columns[0]      # Disease column
feature_cols = df.columns[1:]   # Symptom columns

print("Target Column:", target_col)
print("Total Symptoms:", len(feature_cols))


#EDA
#6.1 Disease Distribution
plt.figure(figsize=(12,5))
df[target_col].value_counts().head(20).plot(kind='bar')
plt.title("Top 20 Most Frequent Diseases")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.show()

#6.2 Symptom Frequency Analysis
symptom_freq = df[feature_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12,4))
symptom_freq.head(20).plot(kind='bar')
plt.title("Top 20 Most Common Symptoms")
plt.xlabel("Symptoms")
plt.ylabel("Frequency")
plt.show()

#6.3 Rare Symptoms
symptom_freq.tail(20)

#6.4 Symptom Correlation (Co-occurrence)
plt.figure(figsize=(15,10))
sns.heatmap(df[feature_cols].corr(), cmap="coolwarm")
plt.title("Symptom Co-occurrence Heatmap")
plt.show()
