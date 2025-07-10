# I skipped Ch 7 because we covered the relevant info in the first chapters. It was just more on data cleaning.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestClassifier

# we use feature selection to identify the most important features in our dataset.
# irrelevant features can add noise and complexity to the model, making it harder to interpret and potentially leading to overfitting.
# for example, if we include features that are not related to the target variable, the model may learn patterns that do not generalize well to new data.
# correlated features cause multicollinearity, which can make it difficult to determine the individual effect of each feature on the target variable.
# for example, if two features are highly correlated, they may provide redundant information, and the model may struggle to determine which one is more important.

# the curse of dimensionality refers to the phenomenon where the performance of a model decreases as the number of features increases, especially when the number of observations is limited.
# basically, if you have a ton of features but not enough data, the model cannot make any meaningful conclusions about the relationships between features and the target variable.

# also, training time is a function of the number of features. more features mean longer training times, which can be a problem for large datasets or complex models.
# we save ourselves time and resources by selecting only the most relevant features.

# load the titanic dataset
df = pd.read_csv('titanic.csv')

# select relevant numeric columns and drop missing values
df = df[['Survived', 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch']].dropna()

# create a new feature: family size
df['FamilySize'] = df['SibSp'] + df['Parch']

# correlation heatmap to visualize feature relationships. this is how we can identify multicollinearity.
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('correlation heatmap')
plt.tight_layout()
plt.show()

# calculate variance inflation factor (VIF) to quantify multicollinearity. this tells us how much the variance of a feature is inflated due to multicollinearity with other features.
# it tells us the harmful features that we should consider removing.

def calculate_vif(X):
    X_const = add_constant(X)
    vifs = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
        index=X_const.columns
    )
    return vifs.drop('const')


# drop features with high VIF iteratively (threshold = 10)
X_clean = df.drop('Survived', axis=1).copy()

print("\ninitial VIF values:")
print(calculate_vif(df.drop('Survived', axis=1)))

while True:
    vifs = calculate_vif(X_clean)
    high_vif = vifs[vifs > 10]
    if high_vif.empty:
        break
    feature_to_drop = high_vif.idxmax()
    print(f"dropping '{feature_to_drop}' with VIF = {vifs[feature_to_drop]:.2f}")
    X_clean = X_clean.drop(columns=[feature_to_drop])

print("\nremaining features after VIF filtering:")
print(X_clean.columns)

# now we can train a random forest classifier to identify important features
X_model = X_clean
y_model = df['Survived']

model = RandomForestClassifier(random_state=42)
model.fit(X_model, y_model)

importances = pd.Series(model.feature_importances_, index=X_model.columns)
important_features = importances.sort_values(ascending=False)

print("\nfinal VIF values:")
print(calculate_vif(X_clean))

print("\nfeature importances (random forest):")
print(important_features)

# you're gonna get infinite VIF values for 'FamilySize' because it is a linear combination of 'SibSp' and 'Parch'.
# I just wanted to show you an example of how to calculate VIF and filter features based on it. In reality, you will observe some VIF values that are high, but not infinite.
# ;)