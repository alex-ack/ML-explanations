import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold

# Load preprocessed Titanic data
train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')

# Split into features and target
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']

# Train a classifier (can be replaced with any model)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# Plot ROC Curve - the ROC curve is a graphical representation of a classifier's performance across different thresholds.
# if the curve is close to the top-left corner, it means the model is good at distinguishing between classes.
# the AUC (Area Under the Curve) quantifies the overall performance of the classifier; a value of 1 indicates perfect classification, while 0.5 indicates random guessing.
def plot_roc_curve(model, X_test, y_test):
    """
    Plots the ROC curve and calculates AUC for a binary classifier.
    """
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability for class 1
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Learning Curve - a learning curve shows how the model's performance changes with different amounts of training data.
# If the training and validation curves converge, it indicates that the model is learning effectively.
# If there's a large gap between the training and validation curves, it suggests overfitting.
# Overfitting is when the model performs well on training data but poorly on unseen data.
# It's normal for this dataset and model since the model isn't tuned.
def plot_learning_curve(model, X, y):
    """
    Plots a learning curve using cross-validation.
    Shows how training and validation scores change with dataset size.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_mean, label="Training Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Call both plotting functions
plot_roc_curve(model, X_test, y_test)
plot_learning_curve(model, X_train, y_train)