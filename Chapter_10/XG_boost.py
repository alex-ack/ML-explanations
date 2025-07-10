import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# this textbook is kind of old. I would use XGBoost package instead of sklearn's GradientBoostingClassifier.
# u need parallel processing so type 'brew install libomp' in terminal

# XGBoost (extreme gradient boosting) is a gradient boosting framework that uses decision trees as base learners.
# gradient boosting is an ensemble technique that builds a model in a stage-wise manner
# by combining weak learners (usually decision trees) to create a stronger predictive model.
# each new tree learns to correct the mistakes made by the previous ones
"""this is how it works:
1.) sequentially adds models that minimize a loss function (e.g. log loss for classification)

2.) each new tree focuses on residual errors from previous trees

3.) xgboost adds regularization (penalty terms) to prevent overfitting

4.) uses advanced optimizations like tree pruning, column subsampling, and parallel computation

"""

# load and prepare the dataset
df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
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

# scale numeric columns
scaler = MinMaxScaler()
X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train xgboost model
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
# # n_estimators: number of trees to build, max_depth: maximum depth of each tree, learning_rate: step size shrinkage to prevent overfitting
xgb.fit(X_train, y_train)

# make predictions
y_pred = xgb.predict(X_test)

# evaluation
print("accuracy:", accuracy_score(y_test, y_pred))
print("\nconfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))