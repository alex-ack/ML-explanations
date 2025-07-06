import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# this script is to show how to save and load a trained model using a python module called pickle.
# load preprocessed training and test data
train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')

# separate features and target
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']

# train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# save the trained model to a file using pickle
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'random_forest_model.pkl'.")

# load the model back from file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# make predictions with the loaded model
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of loaded model: {accuracy:.4f}")
