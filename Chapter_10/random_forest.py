import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🌲🌲🌲🌲🌲🌲

# Random Forest is an ensemble learning method that combines multiple decision trees to improve classification accuracy and control overfitting.
# instead of relying on a single tree... random forest builds many trees (hence "forest") and averages their predictions.

# it builds many decision trees on random subsets of the data (with replacement — bootstrapping)
# selects a random subset of features at each split (adds variety)
# then combines the predictions of all trees to make a final prediction (majority vote for classification, average for regression).

# there is less overfitting compared to a single decision tree because the trees are trained on different subsets of the data and features.

# load and clean data
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
df.dropna(inplace=True)

# encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# scale numerical columns
scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train random forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# make predictions
y_pred = rf.predict(X_test)

# evaluate
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))