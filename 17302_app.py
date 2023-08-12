from flask import Flask, request, render_template,jsonify
import pickle
import numpy as np
import xgboost as xgb

app = Flask(__name__)

with open('xgb_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cust_payment_terms = int(request.form['cust_payment_terms'])
    buisness_year = int(request.form['buisness_year'])
    document_create_date = int(request.form['document_create_date'])
    cust_number = int(request.form['cust_number'])
    is_late_payment = int(request.form['is_late_payment'])
    due_in_date = int(request.form['due_in_date'])

    input_data = np.array([[cust_payment_terms,buisness_year,is_late_payment,document_create_date,due_in_date,cust_number]])
    result = model.predict(input_data)[0]
    predict_json = {'Prediction': result}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)