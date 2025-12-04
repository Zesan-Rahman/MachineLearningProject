from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DIVIDER = "==================================================="
#Importing dataset
data, meta = arff.loadarff("credit-g-dataset.arff")
df = pd.DataFrame(data)
#Getting rid of byte strings
df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
#print(df) #test to see if df works

#Interesting histograms
print(DIVIDER)
print("Histograms")
print(DIVIDER)
for col in df.columns:
    if col in ["duration", "credit_amount","age"]:
        plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribution: {col}")
        plt.xlabel('Values')
        plt.ylabel('Frequencies')
        plt.show()

#Correlation Matrix
print(DIVIDER)
print("Correlation Matrix")
print(DIVIDER)

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#Relationships with target variable
print(DIVIDER)
print("Relationship with target variable")
print(DIVIDER)
target_col = 'class'

# Separate numeric and categorical features
numeric_features = df.select_dtypes(include='number').columns
categorical_features = df.select_dtypes(exclude='number').columns

# Numeric features → boxplot vs class
for feature in numeric_features:
    sns.boxplot(x=target_col, y=feature, data=df)
    plt.title(f'{feature} vs {target_col}')
    plt.show()

# Categorical features → countplot vs class
for feature in categorical_features:
    sns.countplot(x=feature, hue=target_col, data=df)
    plt.title(f'{feature} vs {target_col}')
    plt.show()
