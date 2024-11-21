from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd  # Import pandas to create DataFrame for prediction

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('pass_fail_model.pkl', 'rb'))

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

# Predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract grades from the form
        grade1 = float(request.form['grade1'])  # First grade
        grade2 = float(request.form['grade2'])  # Second grade

        # Validate grades
        if not (0 <= grade1 <= 100 and 0 <= grade2 <= 100):
            raise ValueError("Grades must be between 0 and 100.")

        # Create a DataFrame for prediction with proper feature names
        features = pd.DataFrame([[grade1, grade2]], columns=['Grade1', 'Grade2'])

        # Make prediction
        prediction = model.predict(features)
        result = "Pass" if prediction[0] == 1 else "Fail"

        # Render the result back to the HTML template
        return render_template('index.html', prediction=result)
    except ValueError as e:
        return render_template('index.html', prediction=f"Error: {e}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
# This is a test comment for Heroku deployment

