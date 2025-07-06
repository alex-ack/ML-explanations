import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('train_cleaned_titanic.csv')
test_df = pd.read_csv('test_cleaned_titanic.csv')

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# we trained a simple Random Forest model and made predictions on the test set. now we'll create a confusion matrix to visualize the model's performance.

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# because it's fun, this is how we can visualize the confusion matrix using seaborn and matplotlib.

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
# we create a heatmap of the confusion matrix to see how many true positives, true negatives, false positives, and false negatives our model has.

# commented out - this is scikit-learn's built-in way to display confusion matrices, but we used seaborn for a nicer visualization.
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did Not Survive', 'Survived'])
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()
