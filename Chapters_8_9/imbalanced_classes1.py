import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# you can also use smote (synthetic minority oversampling technique) to oversample the minority class.
# SMOTE generates synthetic samples for the minority class by interpolating between existing samples.
# don't worry about the under/over sampling details in the textbook. I think this is enough.

# load cleaned titanic dataset
df = pd.read_csv('train_cleaned_titanic.csv')

# separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# split into training and testing sets (before applying SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print new class distribution after oversampling
print("class distribution after SMOTE:")
print(y_train_resampled.value_counts())

# train logistic regression model on the resampled training set
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# evaluate on the original test set
y_pred = model.predict(X_test)

print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nclassification report:")
print(classification_report(y_test, y_pred))