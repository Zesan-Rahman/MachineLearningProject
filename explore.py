from scipy.io import arff
import pandas as pd

#Importing dataset
data, meta = arff.loadarff("credit-g-dataset.arff")
df = pd.DataFrame(data)
#Getting rid of byte strings
df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
#print(df) #test to see if df works

#Histogram for numerical features
import matplotlib.pyplot as plt

for col in df.columns:
    if df[col].dtype != 'object':
        plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Distribution: {col}")
        plt.xlabel('Values')
        plt.ylabel('Frequencies')
        plt.show()
