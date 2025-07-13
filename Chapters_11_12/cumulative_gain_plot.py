import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# cumulative gain plots are used to visualize the effectiveness of a classification model, especially in imbalanced datasets.

# they show how well the model can identify positive instances (e.g., survivors in the Titanic dataset) compared to random guessing.

# if we sort the predictions from most confident to least, the cumulative gain curve tells us how many true positives we can expect to find in the top x% of predictions.

# this is really similar to the ROC curve, but instead of plotting true positive rate vs. false positive rate, we plot cumulative gain vs. percentage of samples.

# load data
df = pd.read_csv('titanic.csv')

# preprocess
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# get predicted probabilities for class 1 (survived)
y_proba = model.predict_proba(X_test)[:, 1]

# sort by predicted probability
df_gain = pd.DataFrame({'y_true': y_test, 'y_proba': y_proba})
df_gain = df_gain.sort_values('y_proba', ascending=False).reset_index(drop=True)

# calculate cumulative gains
df_gain['cumulative_positives'] = df_gain['y_true'].cumsum()
df_gain['population_percent'] = np.arange(1, len(df_gain)+1) / len(df_gain)
df_gain['gain_percent'] = df_gain['cumulative_positives'] / df_gain['y_true'].sum()

# plot
plt.figure(figsize=(8, 6))
plt.plot(df_gain['population_percent'], df_gain['gain_percent'], label='model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='baseline')
plt.xlabel('percentage of population')
plt.ylabel('percentage of positives captured')
plt.title('cumulative gain plot (titanic - logistic regression)')
plt.legend()
plt.tight_layout()
plt.show()

# if, for ex., the model captures 50% of the positives in the top 20% of the population, it means that the model is effective at identifying survivors early in the ranked list.

# in ours, the model is very good at ranking survivors near the top, which is exactly what we want in classification problems where we must prioritize top predictions