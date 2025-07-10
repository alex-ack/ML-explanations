import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Naive Bayes is a family of probabilistic algorithms based on Bayes' theorem.
# the naive part refers to the assumption that features are independent given the class label (rarely true in real life).
# this is a strong assumption, but it often works well in practice, especially for text classification tasks.

# baye's theorem is:
# P(A|B) = P(B|A) * P(A) / P(B)
# where:
# P(A|B) is the probability of event A given event B,
# P(B|A) is the probability of event B given event A,
# P(A) is the probability of event A,
# P(B) is the probability of event B.
# in the context of classification, A is the class label and B is the feature vector.
# if a passenger is female, from 1st class, and age < 10, we want to know the probability that they survived based on prior data.
# we write that as: P(Survived∣female, 1st class, age < 10)
# we can avoid full joint probability calculations this way. this is a good website I can't really do it justice here: https://www.ibm.com/think/topics/naive-bayes

# it works well for small datasets and text classification tasks.

"""there is: 

GaussianNB: for continuous data (like age, fare)

MultinomialNB: for discrete counts (like text frequency)

BernoulliNB: for binary features (e.g., 0/1, yes/no)"""

# load and prepare the dataset
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df.dropna(inplace=True)

# encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# normalize continuous variables
scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train naive bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# predict and evaluate
y_pred = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))