import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# logisitic regression is a supervised learning algorithm used for binary classification tasks.
# it predicts the probability that a given input belongs to one of two classes (e.g., survived vs. not survived)
# unlike linear regression (which predicts continuous values), logistic regression predicts a probability between 0 and 1 using the sigmoid function.
# if the probability is greater than 0.5 (usually), the model predicts class 1 (e.g., survived), otherwise it predicts class 0 (e.g., not survived).

# it calculates a weighted sum of input features and passes it through the sigmoid function.

""" this is the sigmoid function:
f(x) = 1 / (1 + exp(-x))
where x is the weighted sum of input features."""

# load data
df = pd.read_csv('titanic.csv')

# clean data
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df.dropna(inplace=True)

# encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# normalize numeric columns
numeric_cols = ['Age', 'Fare', 'FamilySize']
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))