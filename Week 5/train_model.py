import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Creating dataset with two grades
data = {
    'Grade1': [50, 55, 65, 70, 85, 45, 90, 56, 60, 76],  # First grade
    'Grade2': [48, 60, 66, 75, 80, 50, 92, 54, 62, 78],  # Second grade
    'Result': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]  # 1 = Pass, 0 = Fail
}

# Convert it to a DataFrame
df = pd.DataFrame(data)

# Prepare the data
X = df[['Grade1', 'Grade2']]  # Features (two grades)
y = df['Result']   # Labels (Pass/Fail)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file using pickle
with open('pass_fail_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained with two grades and saved as pass_fail_model.pkl")

