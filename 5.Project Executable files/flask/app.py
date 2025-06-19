from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('model/decision_tree_model.pkl', 'rb'))
type_encoder = pickle.load(open('model/type_label_encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/predict")
def predict_page():
    return render_template('predict.html')

@app.route("/pred", methods=["POST"])
def predict():
    try:
        step = float(request.form['step'])
        type_str = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        print(f"Received inputs: {step}, {type_str}, {amount}, {oldbalanceOrg}, {newbalanceOrig}, {oldbalanceDest}, {newbalanceDest}")
        type_encoded = type_encoder.transform([type_str])[0]
        print(f"Encoded type: {type_encoded}")
        input_data = np.array([[step, type_encoded, amount, oldbalanceOrg,
                                newbalanceOrig, oldbalanceDest, newbalanceDest]])

        prediction = model.predict(input_data)[0]
        print(f"Prediction: {prediction}")
        result = "fraudulent" if prediction == 1 else "legitimate"

        return render_template("submit.html", prediction_text=f"The transaction is {result}.")

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
