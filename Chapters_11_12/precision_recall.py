import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# precision recall is a way to evaluate the performance of a binary classifier, especially when dealing with imbalanced classes.
# for example, it is often used in medical diagnosis where the positive class (e.g., disease present) is much rarer than the negative class (e.g., disease absent).
# precision measures the proportion of true positive predictions among all positive predictions made by the model. TP / (TP + FP)
# recall measures the proportion of true positive predictions among all actual positive instances in the dataset. TP / (TP + FN)
# as the decision threshold changes (e.g., from 0 to 1), the precision and recall values change too. The PR curve plots this tradeoff.
# average precision score summarizes the PR curve into a single value, giving you an overall measure of the model's performance across different thresholds.

# use it when you care more about mnimizing false positives or false negatives

# load data
df = pd.read_csv('titanic.csv')

# select features and drop missing values
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = (df['Sex'] == 'male').astype(int)  # encode male as 1

X = df.drop('Survived', axis=1)
y = df['Survived']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

# plot precision-recall curve
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label=f'avg precision = {avg_precision:.2f}')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall curve (titanic - logistic regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
-a high, bowed curve (towards top-right) indicates better performance.

-a horizontal line would represent random guessing.

-you can use this curve to choose a threshold based on the acceptable tradeoff between precision and recall (e.g., if false positives are very costly, favor high precision).
"""

# here, at low recall, the model is very cautious, only predicting someone survived when it’s very sure → high precision.

# as we increase recall (trying to catch more survivors), we also make more false positive mistakes, lowering precision.

# avg is 80, not bad!