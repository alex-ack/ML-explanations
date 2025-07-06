import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report

# stacking is a technique where we combine multiple models to improve performance at the cost of interpretability.
# it is a form of ensemble learning where we use the predictions of multiple base models as input to a final model (the meta-model).
# think of it like a committee of models where each model votes on the final prediction.
# the model learns which base model to trust more based on their performance.
# no free lunch though - sometimes stacking can lead to overfitting, especially if the base models are too complex or if there is not enough data.

train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']
# we separate the features (X) and target (y) for both training and testing sets

base_learners = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
# we define our base learners (the models that will be stacked).

meta_model = LogisticRegression(max_iter=1000, random_state=42)
# we define our meta-model (the model that will learn from the predictions of the base learners).

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,               # 5-fold cross-validation for blending (5 fold means the data is split into 5 parts, and each part is used for training and validation)
    passthrough=False   # If True, raw inputs are passed to meta model too (default is False, meaning only base model predictions are used as input to the meta model)
)

# Train the stacking model
stacking_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacking_clf.predict(X_test)

print("🔷 Stacking Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# it's a bit more accurate now!