import joblib
import pickle

from flask import Flask
from flask import request
from flask import jsonify

dict_vectorizer_file = "dict_vectorizer.pkl"
imputer_file = "imputer.pkl"
model_file = 'XGBoost_2023-11-05_14-54-16.bin'

with open(dict_vectorizer_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(imputer_file, 'rb') as f_in:
    imp = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = joblib.load(model_file)

app = Flask('default')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    print("\nX:", X)
    
    predictions = model.predict_proba(X)
    print("\npredictions:", predictions)
    
    y_pred = predictions[:, 1]
    print("\ny_pred:", y_pred)
    
    default = y_pred >= 0.5
    print("\ndefault:", default)

    result = {
        'creditworthiness': float(y_pred),
        'customer_default': bool(default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
