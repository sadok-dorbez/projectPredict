import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pickle

# Load the data
df = pd.read_csv('African projects Dataset.csv')

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

# Example input to test the predict() method
input_data = {'type': 'tourism',
              'project_cost': 100000000,
              'number_of_workers': 150}

# Convert the input data into a dataframe
input_df = pd.DataFrame([input_data])

# One-hot encode categorical features of the input data
input_df = pd.get_dummies(input_df)

# Make sure the input data has all the columns that the training data has
missing_cols = set(X.columns) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0

# Reorder the columns to match the order in the training data
input_df = input_df[X.columns]

# Make predictions on the input data
y_pred_input = clf.predict(input_df)

# Map the predicted value to class label
y_pred_label_input = le.inverse_transform(y_pred_input)

# Print the predicted value with class label for the input data
print('Predicted value for input:', y_pred_label_input)

# Save the model and the LabelEncoder as pickle files
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('labelencoder.pkl', 'wb') as f:
    pickle.dump(le, f)
