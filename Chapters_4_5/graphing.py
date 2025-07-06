import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# this script shows how to create a few basic plots using seaborn and matplotlib.
# load the titanic dataset
df = pd.read_csv('titanic.csv')

# select relevant columns and drop rows with missing values for clean plotting
df = df[['Age', 'Fare', 'Pclass', 'Survived', 'Sex']].dropna()

# histogram: distribution of a single numerical variable
# this shows the shape of the age distribution (e.g. skewness, peaks). you can see how many mode(s) there are, and if the data is normally distributed.
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('histogram of age')
plt.tight_layout()
plt.show()

# scatter plot: shows relationship between two numerical variables. again, this is easy with pandas.
# here we visualize how fare relates to age, and color by survival
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('scatter plot of age vs fare')
plt.tight_layout()
plt.show()

# joint plot: combines scatter plot and histograms with a regression line
# gives a fuller picture of the relationship between age and fare
sns.jointplot(data=df, x='Age', y='Fare', kind='scatter')
plt.suptitle('joint plot of age and fare', y=1.02)
plt.tight_layout()
plt.show()

# pair plot: shows all pairwise relationships in a dataset. this is a matrix of columns and kernel density estimates (KDE) for each pair of features.
# useful for quick exploratory data analysis across multiple features. you can color by a categorical variable (e.g. survival) to see how it affects the relationships.
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.suptitle('pair plot of selected features', y=1.02)
plt.tight_layout()
plt.show()

# box plot: shows distribution, spread, and outliers grouped by a category
# here we look at fare distributions within each passenger class
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('box plot of fare by passenger class')
plt.tight_layout()
plt.show()

# violin plot: similar to box plot but also shows full distribution shape
# useful when comparing distributions (e.g. age) across survival groups
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='Survived', y='Age')
plt.title('violin plot of age by survival')
plt.tight_layout()
plt.show()