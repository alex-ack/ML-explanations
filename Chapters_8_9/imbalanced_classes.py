import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# imbalanced classes are a common problem in machine learning, especially in classification tasks.
# imbalanced classes occur when the categories in your target variable are not represented equally.
# for example, in the titanic dataset, there are more non-survivors than survivors. the ratio is around 2:1, which isn't terrible.
# sometimes the ratio is more like 100:1, and that is a big problem, because the model will not perform well on the minority class.

# look out for very high accuracy, but terrible precision/recall for the minority class.
# there are a few ways to handle imbalanced classes.

"""1.) One thing we can do is penalize the model for misclassifying the minority class."""

# load the cleaned Titanic dataset
df = pd.read_csv('train_cleaned_titanic.csv')

# split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a logistic regression model with class_weight='balanced'
# this tells the model to penalize errors on minority class more
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# train the model
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)

# evaluate the model
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nclassification report:")
print(classification_report(y_test, y_pred))