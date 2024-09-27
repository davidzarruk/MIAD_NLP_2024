from flask import Flask, jsonify, request
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO
import zipfile

import ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

app = Flask(__name__)

# Load the pre-trained models
models = [joblib.load(f'xgb_model{i}.joblib') for i in range(1, 17)]  # Adjust range based on model count

@app.route('/predict', methods=['GET'])
def predict():
    data_url = request.args.get('url')
    if not data_url:
        return jsonify({'error': 'Missing URL parameter'}), 400

    try:
        print(f"Hola. {data_url}")
        # Use requests to handle the HTTPS URL
        response = requests.get(data_url)
        # Check if the response was successful
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch data'}), 500

        # Assuming the data is in a zip file as per your URL
        with zipfile.ZipFile(BytesIO(response.content)) as thezip:
            with thezip.open(thezip.namelist()[0]) as thefile:
                data = pd.read_csv(thefile)

        # Feature engineering
        data['car_age'] = 2023 - data['Year']
        data.drop(['Year'], axis=1, inplace=True)

        # Collect predictions from each model
        predictions = [model.predict(data) for model in models]

        # Average the predictions
        avg_predictions = np.mean(predictions, axis=0)

        # Prepare results
        results = {'Predictions': [{'ID': int(idx), 'Predicted_Price': float(pred)} for idx, pred in zip(data.index, avg_predictions)]}

        # Return the predictions as JSON
        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0))
    app.run(debug=True)
