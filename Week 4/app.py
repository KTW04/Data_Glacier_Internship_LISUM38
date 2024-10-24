from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('pass_fail_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON input
    score = data['score']  # Extract the student's score
    prediction = model.predict(np.array([[score]]))  # Predict pass/fail
    result = 'Pass' if prediction[0] == 1 else 'Fail'
    return jsonify({'score': score, 'result': result})  # Return the prediction

if __name__ == '__main__':
    app.run(debug=True)
