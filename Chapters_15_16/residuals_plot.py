# I'm skipping chapters 13 and 14 because they are repetitive to much of the content explained earlier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# a residual plot shows the difference between the actual and predicted values of a regression model.

# it helps us understand how well the model fits the data and whether there are patterns in the residuals that indicate issues with the model

# load and prepare data
df = pd.read_csv('titanic.csv')
df = df.dropna(subset=['Age', 'Fare'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# features to predict Fare
X = df[['Pclass', 'Age', 'Sex']]
y = df['Fare']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit linear model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# residual plot
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('predicted fare')
plt.ylabel('residuals')
plt.title('residual plot: predicted fare vs. residuals')
plt.grid(True)
plt.show()

# if the points are randomly scattered around the horizontal line at 0, that's good. it means the model is capturing the relationship well.
# if there are patterns (like a curve), it suggests the model might be missing some important relationships in the data. consider a more complex model (ex., polynomial)