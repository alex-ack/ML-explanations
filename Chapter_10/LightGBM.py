import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

# LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms.
# the difference between LightGBM and other gradient boosting frameworks (like XGBoost) is that LightGBM uses a histogram-based algorithm to speed up training and reduce memory usage.
# it builds trees leaf-wise instead of level-wise, which allows it to handle large datasets more efficiently.
# LightGBM is particularly effective for large datasets and high-dimensional data.
# it works by:
# 1.) converting continuous features into discrete bins (histogram-based)
# 2.) building trees leaf-wise, which means it grows the tree by adding leaves (terminal nodes) rather than levels (branches)
# 3.) it uses gradient boosting one sized 

# load dataset
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
df.dropna(inplace=True)

# encode categorical columns
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# create family size feature
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# scale numeric columns
scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train lightgbm model
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))