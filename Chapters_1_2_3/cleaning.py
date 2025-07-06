import pandas as pd
# we load in the pandas library to work with dataframes. pandas is an easy-to-use data manipulation library in Python.

df = pd.read_csv('titanic.csv')
# we load the titanic dataset from a CSV file into a pandas DataFrame. everyone learns their first ml set-up with this dataset.
# you will see "df" at the beginning of many pandas DataFrame variable names. this is a common convention to indicate that the variable is a DataFrame.
# a dataframe is a 2-dimensional labeled data structure. it can be thought of as a table or a spreadsheet, where each column can be of a different data type (e.g., integers, floats, strings).

df = df.sample(frac=0.2, random_state=42)
# we take a random sample of 20% of the data from the DF. 
# we want to impute only on the training data, not the entire dataset and then use that imputer to fill in the missing values in the test data.
# (imputing is the process of filling in missing values in a dataset).
# otherwise, we are leaking information (cheating by giving future information to the model).

df.head()
# we display the first few rows of the DataFrame to understand its structure and contents. we are checking to make sure the data we have loaded in looks good. always start with this step.
# df also precedes the name of the DataFrame variable, which is a common convention in pandas to indicate that the variable is a DataFrame.

df.info()
# then, we check the DataFrame's info to see the data types and non-null counts of each column. this helps us understand the data better.
# non-null counts tell us how many entries are present in each column. we need to know this to see if there are any missing values (they will need to be dropped or filled in).

df.describe()
# describe gives us a statistical summary of the numerical columns in the DataFrame. this includes count, mean, standard deviation, min, max, and quartiles. it helps us understand the distribution of numerical data.
# BTW, the 'describe' method only works on numerical columns by default. if you want to include categorical columns, you can use df.describe(include='all'), or df.describe(include='object') for just categorical columns.

df.isnull().sum()
# we check for missing values in the DataFrame. this will show us how many null (missing) values are present in each column. we need to handle these missing values before training a model.

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# we are gonna drop the 'PassengerId', 'Name', 'Ticket', and 'Cabin' columns from the DataFrame. these columns are not useful for our analysis or model training.

df['Age'].fillna(df['Age'].median(), inplace=True)
# we fill missing values in the 'Age' column with the median age. this is a common practice to handle missing values in numerical columns. data scientists often use the median to fill in missing values because it is less affected by outliers than the mean.

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# we fill missing values in the 'Embarked' column (the port of embarkation/where they got on) with the mode (most common value). this is a common practice to handle missing values in categorical columns. we use mode becasue it is the most frequent value in the column.

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# we need to convert text tables to numbers. python can't read 'male' or 'female' easily so we  let the numbers 0 and 1 represent them in our file.

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# we convert the 'Embarked' column into dummy variables.
# dummy variables are binary columns that indicate the presence or absence of a category. this is useful for categorical variables in machine learning models.
# python can't read 'C', 'Q', or 'S' easily so we turn them into multiple binary columns (columns for 'C', 'Q', and 'S' where 1 indicates the presence of that category and 0 indicates its absence).

df['FamilySize'] = df['SibSp'] + df['Parch']
# we are creating a brand new column now called 'FamilySize' that is the sum of the 'SibSp' (siblings/spouses aboard) and 'Parch' (parents/children aboard) columns. this gives us a better understanding of the family size of each passenger.

df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
# we need a separate column to indicate if a passenger is alone or not. we create a new column 'IsAlone' that is 1 if the passenger is alone (family size is 0) and 0 otherwise. this can be useful for our analysis or model training.
# you might wonder why we need this column even though we calculated family size in the previous step. well - we might want to use this column as a feature in our model to see if being alone affects survival chances as opposeed to the more granular family size column.

df['Fare'].fillna(df['Fare'].median(), inplace=True)
# we fill missing values in the 'Fare' column with the median fare. this is a common practice to handle missing values in numerical columns, similar to how we handled the 'Age' column.

print(df.isnull().sum())
# we check again for missing values in the DataFrame to ensure that all missing values have been handled. this is a good practice to confirm that our data cleaning steps were successful.

print(df.head())
# we preview the final dataset to make sure it looks OK.

df.to_csv('sample_cleaned_titanic.csv', index=False)
# we save the cleaned DataFrame to a new CSV file called 'cleaned_titanic.csv'. we set index=False to avoid writing row indices to the file.
# that just means we don't want to write the row numbers to the file, we just want the data itself.

""" The sample data is clean now! We can use it for other fun things."""