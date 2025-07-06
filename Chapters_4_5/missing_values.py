import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the original titanic dataset (not yet cleaned)
df = pd.read_csv('titanic.csv')

# overview of missing data
print("missing values per column:\n")
print(df.isnull().sum())

print("\npercentage of missing values:\n")
print((df.isnull().sum() / len(df) * 100).round(2))

# sometimes we need more specific information about missing values, such as the percentage of missing values per column.
# heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("missing data heatmap")
plt.tight_layout()
plt.show()

# a bar plot is also a good way to visualize missing data.
# bar plot of missing value percentages
missing_percent = (df.isnull().sum() / len(df)).sort_values(ascending=False)
missing_percent = missing_percent[missing_percent > 0]

plt.figure(figsize=(8, 4))
missing_percent.plot(kind='bar', color='salmon')
plt.ylabel('fraction of missing values')
plt.title('missing data per column')
plt.tight_layout()
plt.show()

""" 
handling missing values will depend on the context and the specific dataset.
common strategies include:
1. Dropping rows or columns with missing values
2. Filling missing values with a specific value (e.g., mean, median, mode)
3. Using interpolation or extrapolation methods
4. Using more machine learning models to predict missing values
use the strategy that makes the most sense for your data and analysis goals.
""" 

# here we will just drop them. handle missing values:
df_dropped = df.drop(['Cabin'], axis=1)

df_dropped['Age'].fillna(df_dropped['Age'].median(), inplace=True)
df_dropped['Embarked'].fillna(df_dropped['Embarked'].mode()[0], inplace=True)

# check again for missing values
print("\nremaining missing values after cleaning:\n")
print(df_dropped.isnull().sum())

# save the cleaned dataset
df_dropped.to_csv('titanic_missing_handled.csv', index=False)
print("\ncleaned dataset saved as 'titanic_missing_handled.csv'.")
