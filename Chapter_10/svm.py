import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# svm (support vector machine) is a supervised learning algorithm for classification tasks.
# given labeled data (e.g., survived vs. not survived), svm tries to:
# 1.) find the best hyperplane that separates the classes in the feature space.
# 2.) maximize the margin between the hyperplane and the closest data points from each class (support vectors).
# svm can handle both linear and non-linear classification tasks by using different kernel functions.

# a hyperplane is just a line (in 2d), a plane (in 3d), or a higher-dimensional separator.
# it's the decision boundary that separates the classes: 0 meaning predict class 0 (ex. not survived), 1 meaning predict class 1 (ex. survived).

# the margin is the distance between the hyperplane and the nearest points from each class.
# svm tries to maximize this margin, which helps improve generalization and reduce overfitting.

# how it works:
""" 1.)plot all your points in space
    2.) find the possible lines (hyperplanes) that separate the classes
    3.) find the line that maximizes the margin between the classes
"""

# it works well for high-dimensional data and can handle non-linear relationships using kernel functions.

# but, not ideal for very large datasets, and choosing the kernel can get tricky 

# load dataset
df = pd.read_csv('titanic.csv')

# select relevant columns and drop missing values
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
df.dropna(inplace=True)

# encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# create new feature
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# scale numeric features
numeric_cols = ['Age', 'Fare', 'FamilySize']
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')  # 'rbf' is good for nonlinear boundaries
svm_model.fit(X_train, y_train)

# make predictions
y_pred = svm_model.predict(X_test)

# evaluate the model
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))