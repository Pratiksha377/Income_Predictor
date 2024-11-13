import os
import pandas as pd

# Create project folders
os.makedirs('Income_Prediction_Project/data', exist_ok=True)

# Download and save the dataset (preprocessed version)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income'
]

# Load data directly from the URL, if you're online (or save it as 'adult_income.csv' locally)
data = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
data.to_csv('Income_Prediction_Project/data/adult_income.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('../data/adult_income.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Display the first few rows
data.head()
# Convert categorical features to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop(columns=['income_>50K'])
y = data['income_>50K']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
