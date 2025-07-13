import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# a learning curve shows how the model's performance changes as the size of the training data increases.
# it helps us understand how much data we need to train the model effectively and whether the model is overfitting or underfitting.

# load the titanic dataset
df = pd.read_csv('titanic.csv')

# basic preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

X = df.drop('Survived', axis=1)
y = df['Survived']

# decision tree classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=5),
    scoring='accuracy',
    shuffle=True,
    random_state=42
)

# compute average scores
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='training score')
plt.plot(train_sizes, val_mean, label='validation score')
plt.xlabel('training set size')
plt.ylabel('accuracy')
plt.title('learning curve (titanic - decision tree)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#if the gap stays large, the model might be overfitting

# if both curves flatten early and stay low, the model might be underfitting

# the training score starts high and the validation score starts low, but they converge as the training size increases.

# this is expected. With fewer examples, the model overfits the small dataset and achieves high accuracy.

# as more data is added, the model has to generalize more and can't memorize, so training accuracy drops slightly.

# with very few training samples, the model performs poorly on validation data.

# as you give it more data, validation accuracy improves and stabilizes.

# the gap remains pretty big, which suggests the model is overfitting.

# the model is learning patterns specific to the training data that don’t generalize well to new data

# you could: try simplifying the model (e.g. limit tree depth more), regularize, use more data, or use ensemble methods like random forests to reduce overfitting