import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# heteroscedasticity refers to the situation where the variance of the residuals is not constant across all levels of the independent variable(s), and we need further probing
# this can lead to inefficient estimates and affect the validity of statistical tests.
# to check for heteroscedasticity, we can use several methods:
# 1. residual plot: plot residuals against predicted values to see if they fan out or show a pattern.
# 2. histogram of residuals: check if the distribution of residuals is bell-shaped.
# 3. Q-Q plot: compare the quantiles of the residuals to the quantiles of a normal distribution. ()


# load titanic data
df = pd.read_csv('titanic.csv')
df = df.dropna(subset=['Age', 'Fare'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# select features and target
X = df[['Pclass', 'Age', 'Sex']]
y = df['Fare']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# predict and calculate residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# 1. residual plot (heteroscedasticity check)
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('predicted fare')
plt.ylabel('residuals')
plt.title('residual plot (check for heteroscedasticity)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. histogram (check for bell curve shape)
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, bins=30)
plt.title('histogram of residuals')
plt.xlabel('residual')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()

# 3. Q-Q plot (check if points fall on line)
plt.figure(figsize=(6, 4))
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q plot of residuals')
plt.tight_layout()
plt.show()

# 4. normality test (e.g. D’Agostino and Pearson)
stat, p = stats.normaltest(residuals)
print(f"normality test p-value: {p:.4f}")
if p > 0.05:
    print("residuals are likely normal")
else:
    print("residuals are likely not normal")

# you will see that the residuals are not normally distributed, which indicates that the model may not be capturing all the relationships in the data.
# in the titanic dataset, that is because the relationship between the features and the target variable (Fare) is not linear.
# many passengers (especially in 3rd class or children) had a Fare of 0. so this is not a good dataset to use to demonstrate normality of residuals. but this is the process.