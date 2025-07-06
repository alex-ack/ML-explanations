import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('sample_cleaned_titanic.csv')
# we load our new cleaned dataset in.
# we are going to normalize our data. data normalization is the process of scaling numerical data to a specific range, usually between 0 and 1, or -1 and 1.
# this is useful for machine learning algorithms that are sensitive to the scale of the data, such as gradient descent-based algorithms.
# it is important to avoid biasing the model by having features with different scales. for example, if one feature has a range of 0-100 and another has a range of 0-1, the model might give more importance to the first feature just because it has a larger range.


df.head()
# we display the first few rows of the cleaned DataFrame to understand its structure and contents after cleaning.

numeric_cols = ['Age', 'Fare', 'FamilySize']
# we define a list of numeric columns that we want to normalize. these are the columns that contain numerical data.
# these contain continuous numerical data that we want to scale to a similar range.
# we do not include binary or categorical columns here, as they do not need normalization.

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# now we scale the data to a range of 0 to 1 using MinMaxScaler from scikit-learn.
# This rescales each numeric column to range [0, 1]

df.to_csv('normalized_sample.csv', index=False)
# we'll save a new file now with the normalized data.
# I separated the scripts for clarity, but this could be done in the same script as the cleaning step if you prefer.
# there is a sleeker, combined script called "clean_and_normalize.py" that does both steps in one go.