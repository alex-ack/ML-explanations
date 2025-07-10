import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# K-Nearest Neighbors (KNN) is a simple, non-parametric classification algorithm.
# it makes predictions by comparing a new data point to the k closest points in the training set.
# it assigns the most common class (majority vote) among those neighbors.

"""how it works:
1.) choose a value for k (number of neighbors to consider) (ex. k=5)
2.) for a new data point, calculate the distance to all points in the training set (usually using Euclidean distance)
3.) find the k nearest neighbors (closest points)
4.) for classification, assign the class that is most common among those k neighbors.
"""

# KKN is good for small to medium sized datasets and works well with non-linear decision boundaries.
# however, it is slow for large datasets because it needs to calculate distances to all points in the training set for each prediction.
# it's also senstive to imbalanced classes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# scale numeric columns
numeric_cols = ['Age', 'Fare', 'FamilySize']
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # k = 5
knn.fit(X_train, y_train)

# make predictions
y_pred = knn.predict(X_test)

# evaluate
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
# Note: You can adjust the value of k (n_neighbors) to see how it affects the model's performance.