import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# train_test_split lets us split data into training and testing sets in one line.
# MinMaxScaler helps normalize numerical features to a range between 0 and 1.

# --------------------------------------------------------
# STEP 1: Define a function to clean the Titanic dataset
# --------------------------------------------------------

"""'def' is what we use to define a function in Python. 
    functions are reusable blocks of code that perform a specific task."""

def clean_titanic_data(df):
    """
    cleans the Titanic dataset by:
    - dropping irrelevant or sparse columns
    - handling missing values (imputation)
    - encoding categorical variables
    - creating new, useful features (feature engineering)

    parameters:
    df (DataFrame): Raw Titanic data

    returns:
    DataFrame: cleaned DataFrame ready for modeling
    """

    # drop irrelevant or high-missing columns
    # 'Cabin' is mostly empty; 'Name', 'Ticket', and 'PassengerId' don't help predict survival
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)        # median is less affected by outliers
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # mode is best for categorical

    # convert 'Sex' from text to binary numeric (0 = male, 1 = female)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # convert 'Embarked' into dummy variables (one-hot encoding)
    # drop_first=True prevents multicollinearity by dropping the first category (basically, it allows us to avoid the dummy variable trap) (dw about it right now)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # feature engineering:
    # create a new column for family size by combining siblings/spouse and parents/children
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # create a binary column 'IsAlone': 1 if passenger is alone, 0 otherwise
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

    # return the cleaned DataFrame
    return df

# --------------------------------------------------------
# STEP 2: Define a function to split and normalize the data
# --------------------------------------------------------

def prepare_train_test_sets(df, test_size=0.2):
    """
    splits cleaned data into training and testing sets,
    and normalizes numeric features using only the training set to avoid data leakage.

    parameters:
    df (DataFrame): cleaned titanic dataset
    test_size (float): Fraction of data to use for testing (default is 20%)

    returns:
    train (DataFrame): rraining data with normalized numeric features and target
    test (DataFrame): rest data with normalized numeric features and target
    """

    # separate features (X) and target (y)
    X = df.drop('Survived', axis=1)  # All columns except 'Survived'
    y = df['Survived']               # The target we want to predict

    # Split the dataset into training and testing sets
    # random_state ensures reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # normalize only the numeric columns using MinMaxScaler
    # this ensures all values are scaled between 0 and 1
    numeric_cols = ['Age', 'Fare', 'FamilySize']
    scaler = MinMaxScaler()

    # fit the scaler ONLY on the training data, then apply to both train and test sets
    # this prevents data leakage (accidentally cheating and letting test data influence training)
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # reattach the 'Survived' target column to each set for convenience
    train = X_train.copy()
    train['Survived'] = y_train

    test = X_test.copy()
    test['Survived'] = y_test

    return train, test

# --------------------------------------------------------
# STEP 3: Execute the cleaning and preprocessing pipeline
# --------------------------------------------------------

# load the raw Titanic dataset
raw_df = pd.read_csv('titanic.csv')  # assumes the CSV is in the same directory

# apply cleaning function
cleaned_df = clean_titanic_data(raw_df)

# split into train/test sets and normalize numeric features
train_set, test_set = prepare_train_test_sets(cleaned_df)

# save the final processed sets to CSV
train_set.to_csv('train_cleaned_titanic.csv', index=False)
test_set.to_csv('test_cleaned_titanic.csv', index=False)

# you now have clean, normalized training and testing data ready for modeling.
