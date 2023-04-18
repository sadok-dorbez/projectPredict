import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the data
df = pd.read_csv('./African projects Dataset.csv')

# Encode the target variable
le = LabelEncoder()
df['project_result'] = le.fit_transform(df['project_result'])

# One-hot encode categorical features
df = pd.get_dummies(df)

# Split the data into training and testing sets
X = df.drop('project_result', axis=1)
y = df['project_result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Print the accuracy
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)


# Print the feature names
feature_names = X.columns.tolist()
print('Feature names:', feature_names)

# Map the predicted values to class labels
y_pred_labels = le.inverse_transform(y_pred)

# Print the predicted values with class labels
print('Predicted values:', y_pred_labels)

# Save the model and the LabelEncoder
joblib.dump(clf, 'model.joblib')
joblib.dump(le, 'labelencoder.joblib')
