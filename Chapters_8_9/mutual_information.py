from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# mutual information is a measure of the amount of information that one random variable contains about another random variable.
# if two variables are independent, their mutual information is zero.

df = pd.read_csv('train_cleaned_titanic.csv')

# assuming df is already cleaned and processed
X = df.drop('Survived', axis=1)
y = df['Survived']

# compute mutual information
mi_scores = mutual_info_classif(X, y, discrete_features='auto')

# make results readable
mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

print("mutual information scores:")
print(mi_series)

# sex is the most informative feature (.160730), followed by fare and Pclass.