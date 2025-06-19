from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained Random Forest model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_form():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def predict():
    try:
        # Collect form inputs from the form
        step = float(request.form['step'])
        tx_type = request.form['tx_type'].strip().upper()  # fixed here
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud= float(request.form['isFlaggedFraud'])
        # Encode transaction type (update as per your model)
        type_mapping = {
            "TRANSFER": 0,
            "CASH_OUT": 1
        }

        if tx_type not in type_mapping:
            return render_template('submit.html', prediction_text="❌ Invalid transaction type")

        type_encoded = type_mapping[tx_type]

        # Prepare data for prediction
        isFlaggedFraud= float(request.form['isFlaggedFraud'])
        x = np.array([[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,isFlaggedFraud]])

        # Make prediction
        pred = model.predict(x)[0]
        result = "✅ FRAUD DETECTED" if pred == 1 else "✅ NO FRAUD DETECTED"

        return render_template('submit.html', prediction_text=result)

    except Exception as e:
        return render_template('submit.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
