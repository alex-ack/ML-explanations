import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import radviz, parallel_coordinates

# I'm leaving a few more examples of visualizations here that are useful for exploratory data analysis.
# load the titanic dataset
df = pd.read_csv('titanic.csv')

# select relevant features and drop rows with missing values
df = df[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived']].dropna()

# ordinal comparison: compare survival rates across passenger classes. ordinal variables are variables that have a *meaningful* order but no fixed distance between them.
# for ex., class is ordinal because there is a meaningful order (1st, 2nd, 3rd) but the difference between classes is not fixed (e.g. 1st class is not necessarily twice as good as 2nd class).

"""survival is kind of weird. We are treating it as ordinal (there is a natural order: surviving is better than not)
but it is actually binary and has only two values (0 or 1) which in most scenarios would be treated as categorical."""

# this shows average survival by class, since both are ordinal
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Pclass', y='Survived', estimator=lambda x: sum(x) / len(x))
plt.title('average survival by passenger class')
plt.tight_layout()
plt.show()

# correlation heatmap: show strength and direction of linear relationships
corr = df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('correlation heatmap')
plt.tight_layout()
plt.show()

# radviz plot: project multivariate data into 2d based on radial forces
# each feature pulls a point toward itself depending on its value
df_radviz = df.copy()
df_radviz['Survived'] = df_radviz['Survived'].astype(str)  # needed for coloring
plt.figure(figsize=(6, 6))
radviz(df_radviz, 'Survived')
plt.title('radviz plot by survival')
plt.tight_layout()
plt.show()

# parallel coordinates plot: each feature is a vertical axis
# each line represents one passenger and crosses all features
plt.figure(figsize=(8, 5))
parallel_coordinates(df_radviz, 'Survived', color=['#1f77b4', '#ff7f0e'])
plt.title('parallel coordinates plot by survival')
plt.tight_layout()
plt.show()