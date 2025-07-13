import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# a validation curve plots model performance (usually accuracy, f1, etc.) against different values of a specific hyperparameter
# this helps us understand how the model's performance changes as we vary the hyperparameter, allowing us to choose the best value for it.
# it shows both:
    #training score: how well the model fits the training data
    #validation score: how well it generalizes to unseen data
# so, you want a hyperparameter that gives a high validation score without overfitting to the training data (high training score).

# load pre-cleaned titanic data
df = pd.read_csv('train_cleaned_titanic.csv')

# separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# define model
model = DecisionTreeClassifier(random_state=42)

# test different values of max_depth (tree depth)
param_range = range(1, 21)

# generate training and validation scores
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# compute mean scores across folds
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# plot the results
plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label="training score")
plt.plot(param_range, val_mean, label="validation score")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("validation curve for decision tree (titanic)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# choose the number of params where validation peaks (should be 3-4 here)