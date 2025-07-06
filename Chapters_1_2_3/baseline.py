import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

# this is a baseline model. baseline models are simple models that we use to compare against more complex models. they help us understand if our complex models are actually improving performance or not.

train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')
# we load the data

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']
# we separate the features (X) and target (y) for both training and testing sets
# all this does is create two dataframes.
# X_train and X_test contain all columns except 'Survived'.

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)
# we create a dummy classifier (a simple model that predicts the most frequent class) and fit it to our training data.

print("Dummy Classifier (Majority Class Baseline)")
print("Accuracy:", accuracy_score(y_test, y_dummy_pred))
print(classification_report(y_test, y_dummy_pred))
# we evaluate the dummy classifier's performance using accuracy and a classification report, which includes precision, recall, and F1-score for each class.

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# we'll create a logistic regression model, fit it to our training data, and make predictions on the test data.

print("\n Logistic Regression (Smart Baseline)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# same as before, we report accuracy, precision, recall, and F1-score for the logistic regression model.