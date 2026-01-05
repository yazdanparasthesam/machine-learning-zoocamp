import pickle
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('bank-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    subscribe = y_pred >= 0.5

    result = {
        'subscription_probability': float(y_pred),
        'subscribe': bool(subscribe)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)