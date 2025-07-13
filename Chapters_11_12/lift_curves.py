import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# load Titanic dataset (you can replace this with your version)
# lift curves are used to evaluate the performance of classification models, especially in binary classification tasks.
# the lift curve plots the ratio of true positives captured by the model against the percentage of the population considered.

df = pd.read_csv('titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df.dropna(inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# predict probabilities
y_scores = model.predict_proba(X_test_scaled)[:, 1]

# create lift curve data
def plot_lift_curve(y_true, y_scores):
    # Sort by predicted scores
    data = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
    data.sort_values(by='y_scores', ascending=False, inplace=True)

    data['cumulative_positives'] = data['y_true'].cumsum()
    total_positives = data['y_true'].sum()
    population = np.arange(1, len(y_true) + 1)

    lift = (data['cumulative_positives'] / population) / (total_positives / len(y_true))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(population / len(y_true), lift, label='Lift curve', color='blue')
    plt.hlines(1, 0, 1, colors='gray', linestyles='dashed', label='Random baseline')
    plt.xlabel('Percentage of population')
    plt.ylabel('Lift')
    plt.title('Lift Curve (Titanic - Logistic Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# call function
plot_lift_curve(y_test.values, y_scores)

# the lift curve usually bows upwards, indicating that the model is effective at identifying positive instances (e.g., survivors in the Titanic dataset) compared to random guessing.

"""at the very beginning, your model has a lift of around 2.4. that means the top few percent of our predictions capture 2.4× more positives than random guessing

as we go further right (include more of the population), lift gradually drops this is normal: we're adding lower-confidence predictions

by the time we're at 100% of the population, lift = 1, same as random guessing — you’ve included everyone now

"""