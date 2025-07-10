import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# now I'm going to show you how to use Lasso regression, which is a linear regression model that uses L1 regularization to prevent overfitting by penalizing large coefficients.
# L1 regularization is a technique that adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function.
# so, if a feature is irrelevant, Lasso will shrink its coefficient to zero, effectively removing it from the model.
# look at the diagram it helps

# load cleaned titanic dataset
df = pd.read_csv('train_cleaned_titanic.csv')

# standardize the fetures (need to do this because Lasso is sensitive to the scale of the features)
X = df.drop('Survived', axis=1)
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)

# train a lasso regression model
lasso = Lasso(alpha=0.05)  # smaller alpha = less shrinkage
lasso.fit(X_train, y_train)

# view coefficients
coef = pd.Series(lasso.coef_, index=X.columns)
print("lasso coefficients:")
print(coef)

# print non-zero features
print("\nfeatures selected by lasso (non-zero coefficients):")
print(coef[coef != 0])

# evaluate performance (optional for regression-like output)
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nmean squared error: {mse:.4f}")

# plot coefficients
plt.figure(figsize=(8, 4))
coef.plot(kind='bar')
plt.title('lasso regression coefficients')
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

# a coefficient of zero means that the feature was not selected by Lasso, while a non-zero coefficient means that the feature was selected and its importance is reflected in the magnitude of the coefficient.
# Lasso regression is particularly useful when you have a large number of features and want to perform feature selection automatically.
# lasso regression is a regression model, so it is best suited for continuous target variables. However, it can also be adapted for classification tasks by using logistic regression with L1 regularization.
# Note: Lasso regression is sensitive to the choice of alpha (the regularization strength). You may want to experiment with different values of alpha to find the best model for your data.
# google it

# the graph will show nonzero values for sex and class.
# because chivalry&rich people having safer/higher room positioning, women were more likely to survive and lower classes were less likely to survive.
# that is why their coefficients are non-zero - they are important features in predicting survival.