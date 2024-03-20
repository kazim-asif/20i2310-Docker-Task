from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained model
rf_classifier = joblib.load('random_forest_model.joblib')

@app.route('/', methods=['GET'])
def root():
    return jsonify({'response': 'This is the main/default route'})

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json

    # Preprocess input data
    input_data = pd.DataFrame(data)
    input_data['Sex'] = label_encoder.transform(input_data['Sex'])
    input_data['Embarked'] = label_encoder.transform(input_data['Embarked'])

    # Make predictions
    predictions = rf_classifier.predict(input_data)

    # Return predictions
    return jsonify({'predictionsList': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True) #debuger is active