import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# recursive Feature Elimination with Cross-Validation (RFECV) is a feature selection method that recursively removes features and builds a model on the remaining features.
# it's kind of like lasso regression, but instead of shrinking coefficients, it removes features entirely.
# it trains the model multiple times, each time removing the least important features based on the model's performance.

# load cleaned titanic dataset
df = pd.read_csv('train_cleaned_titanic.csv')

# separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)

# define model
model = LogisticRegression(max_iter=1000)

# create RFECV selector
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

# print results
print("optimal number of features:", rfecv.n_features_)
print("selected features:", X.columns[rfecv.support_].tolist())

# plot performance vs number of features
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
# textbook says this, it's depreciated, though: plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel("number of features selected")
plt.ylabel("cross-validation score (accuracy)")
plt.title("RFECV performance vs number of features")
plt.tight_layout()
plt.show()

# you should see the perfromance curve peaking at a certain number of features.
# this indicates the optimal number of features to use for the model.
# it should be 5 - it peaks at around 80% accuracy.

# get the mask of selected features
selected_mask = rfecv.support_

# get the actual feature names
selected_features = X.columns[selected_mask]

print("Selected features:")
print(selected_features.tolist())

# this prints the feature names that were selected by RFECV that help us the most.