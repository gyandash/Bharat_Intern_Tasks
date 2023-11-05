import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Analyze the dataset
titanic.head()

# Data preprocessing
titanic.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)
titanic.dropna(inplace=True)

# Convert categorical variables to numerical values
titanic = pd.get_dummies(titanic, columns=['sex', 'class', 'embarked'], drop_first=True)

# Define features and target
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
