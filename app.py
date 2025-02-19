# app.py
from flask import Flask, request, render_template
import numpy as np
import joblib

# Load model and metadata
model = joblib.load("model/House_price_prediction.joblib")
training_feature_means = np.load("model/feature_means.npy")

# Initialize Flask app
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        MedInc = int(request.form['MedInc'])
        HouseAge = int(request.form['HouseAge'])
        AveRooms = int(request.form['AveRooms'])
        AveBedrms = int(request.form['AveBedrms'])
        Population = int(request.form['Population'])
        AveOccup = int(request.form['AveOccup'])
        Latitude = int(request.form['Latitude'])
        Longitude = int(request.form['Longitude'])
        

        # Create input data for prediction
        input_data = [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]

        # Make prediction using the trained model
        price = model.predict(input_data)[0]

        # Return the prediction result
        result = price
        return render_template('result.html', prediction=result)

if __name__== "__main__":
    app.run(debug=True)
