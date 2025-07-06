import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# this is a Random Forest model. Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and control overfitting.

train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']

rf_model = RandomForestClassifier(
    n_estimators=100,       # Number of trees in the forest
    max_depth=None,         # Let trees grow until pure leaves
    random_state=42         # For reproducibility
)

rf_model.fit(X_train, y_train)

"""-Number of trees just means how many decision trees we want to build in our forest. More trees usually mean better performance, but also more computation time and memory usage.
   -Max depth controls how deep each tree can grow. If None, trees can grow until all leaves are pure or contain less than min_samples_split samples. Limiting depth can help prevent overfitting.
   -That sounds weird, but it just means we want the model to be able to learn complex patterns without memorizing the training data too much and overgeneralizing to new data.
   -Random state is just for reproducibility, so we can get the same results every time we run the code."""

# sci-kit makes it easy

y_pred = rf_model.predict(X_test)

print("Random Forest Classifier Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# this is how we print our results.