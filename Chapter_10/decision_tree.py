import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import tree

# 🌳🌳🌳🌳🌳🌳
# decision trees are a type of supervised learning algorithm used for classification and regression tasks.

""""a decision tree is a tree-like structure where:
1.) each internal node represents a feature (attribute) of the data
2.) each branch represents a decision based on that feature
3.) each leaf node represents a final prediction

the tree is built by recursively splitting the data based on the feature that provides the best separation of classes (for classification tasks) or minimizes the error (for regression tasks).
"""
# it's called a tree because it has a root node (the top node), branches (the splits), and leaf nodes (the final predictions). look @ the textbook diagram

"""how it works:
1.) start with the entire dataset at the root node
2.) for each internal node, select the feature that best separates the classes:
     classification: uses gini impurity or entropy to choose the best split
     regression: uses mean squared error (mse)
3.) split the data into subsets based on the selected feature"""

# decision trees are good for both classification and regression tasks.
# size of the dataset doesn't matter much, but they can overfit if the tree is too deep.

# load and prepare the data
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
df.dropna(inplace=True)

# encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# create new feature
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# scale numeric columns
scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train decision tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)  # limit depth to prevent overfitting
clf.fit(X_train, y_train)

# make predictions
y_pred = clf.predict(X_test)

# evaluate model
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))

# visualize tree
plt.figure(figsize=(16,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("decision tree")
plt.show()

# use max_depth, min_samples_split, or min_samples_leaf to prevent overfitting