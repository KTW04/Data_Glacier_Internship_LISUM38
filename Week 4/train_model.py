import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Creating dataset
data = {
    'Score': [50, 55, 65, 70, 85, 45, 90, 56, 60, 76],  # Student scores
    'Result': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]  # 1 = Pass, 0 = Fail
}

# Convert it to a DataFrame
df = pd.DataFrame(data)

# Prepare the data
X = df[['Score']]  # Features (Student scores)
y = df['Result']   # Labels (Pass/Fail)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a file using pickle
with open('pass_fail_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as pass_fail_model.pkl")
