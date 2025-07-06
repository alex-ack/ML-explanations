import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# This script is used to find the best hyperparameters for a Random Forest model using Grid Search.
# Hyperparameters tell us how to build our model, like how many trees to use, how deep they can grow, etc.

def load_data():
    """
    Load preprocessed titanic training and testing datasets.
    Returns:
        X_train, y_train, X_test, y_test
    """
    train_df = pd.read_csv('train_cleaned_titanic.csv')
    test_df = pd.read_csv('test_cleaned_titanic.csv')
    # and split the data into features (X) and target (y)
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']
    return X_train, y_train, X_test, y_test
# and return it so you can use it later.
def tune_random_forest(X_train, y_train):
    """
    Perform grid search to optimize hyperparameters for a Random Forest classifier. Built in feature of scikit.
    Returns:
        grid_search: the fitted GridSearchCV object
    """
    model = RandomForestClassifier(random_state=42)
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]     # Minimum samples required to be at a leaf node
    } # :D
    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # Use accuracy as the metric to optimize
        verbose=1,  # Print progress messages
        n_jobs=-1  # Use all available cores
    ) # grid search will try all combinations of hyperparameters in the grid and evaluate them using cross-validation.
    grid_search.fit(X_train, y_train)  # Fit the grid search to the training data
    return grid_search  # Return the fitted grid search object with the best parameters found

# now we will make a new function to evaluate the best model found by grid search

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print accuracy and classification report.
    Parameters:
        model: the fitted model to evaluate
        X_test: features of the test set
        y_test: true labels of the test set
    """
    y_pred = model.predict(X_test)  # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print("Best model performance on test set:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# now we need to add something that will run the whole process. we can use the if __name__ == "__main__": block to do that.

if __name__ == "__main__":
    # Load the data
    X_train, y_train, X_test, y_test = load_data()
    
    # Tune the Random Forest model
    grid_search = tune_random_forest(X_train, y_train)
    
    # Print the best hyperparameters found
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    
    # Evaluate the best model on the test set
    evaluate_model(grid_search.best_estimator_, X_test, y_test)