import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.preprocessing import MinMaxScaler

# TPOT (Tree-based Pipeline Optimization Tool) is an automated machine learning library that optimizes machine learning pipelines using genetic programming.
# genetic programming is a technique that evolves programs (in this case, machine learning pipelines) over generations to find the best solution.
# it searches through a space of possible pipelines, combining and mutating them to find the best-performing one.

# load and preprocess the data
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
df.dropna(inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialize and run TPOT
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1)
tpot.fit(X_train, y_train)
# generation tells TPOT how many iterations to run, population_size is the number of pipelines in each generation, and verbosity controls the ouptut level (2 is quite verbose).
# n_jobs=-1 allows TPOT to use all available CPU cores for parallel processing

# evaluate
print("tpot test accuracy:", tpot.score(X_test, y_test))

# export the best pipeline
tpot.export('best_pipeline_titanic.py')