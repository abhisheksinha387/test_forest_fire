import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

# Load Ridge Regressor and Standard Scaler pickle
ridge_model = pickle.load(open('./models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get inputs from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
        except ValueError:
            # Handle invalid inputs
            return render_template('home.html', results="Invalid input. Please enter numeric values.")

        # Combine input data into a list
        data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

        # Standardizing the data
        data_scaled = standard_scaler.transform(data)

        # Predicting the result
        result = ridge_model.predict(data_scaled)[0]

        # Render the result to the template
        return render_template('home.html', results=result)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
