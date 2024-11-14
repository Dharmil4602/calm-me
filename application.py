import numpy as np
import pandas as pd
from models.models import load_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from database.db import Connection
from uuid import uuid1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
if __name__ == "__main__":
    app.run(debug=True)
    
db = Connection(os.getenv("MONGODB_DB"))
scaler = StandardScaler()

# Flask API
@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    # Checking if the data is a list or a single object
    if not isinstance(data, list):
        data = [data]        
    dataFromFunction = loadHrvInModel(data)
    
    register_user(dataFromFunction)
    return {"message": dataFromFunction}

def loadHrvInModel(data):
    results = []
    hrvData_array = []
    for i in data:
        for hrvData in i['healthData']:
            hrv = hrvData['value']
            email = i['email']
            dateFrom = hrvData['dateFrom']
            dateTo = hrvData['dateTo']
            
            input_data = np.asarray([[hrv]])
            dataset = pd.read_csv('./data/sdnn_labeled_dataset.csv')
            hrv_dataframe = pd.DataFrame(dataset, columns=['hrv_val'])
        
            scaler.fit(hrv_dataframe)
            input_data_scaled = scaler.transform(input_data)
            
            model_name = 'svm'
            
            model = load_model(model_name)
            prediction = model.predict(input_data_scaled)
            
            stress_mapping = {0: "Chronic Stress", 1: "Low Stress", 2: "Mild Stress"}
            prediction_result = stress_mapping.get(int(prediction[0]), "Unknown")
            label = 0 if prediction_result == "Chronic Stress" else 1 if prediction_result == "Low Stress" else 2
            
            hrvData_array.append({
                "hrv": hrv,
                "prediction": prediction_result,
                "label": label,
                "dateFrom": dateFrom,
                "dateTo": dateTo
            })
            results.append({
                "email": email,
                "stress_data": hrvData_array
            })
    return results

def register_user(data=None):
    if isinstance(data, list):
        for item in data:
            item["_id"] = str(uuid1().hex)
        result = db['prediction_results'].find_one_and_update({'email': item['email']}, {'$push': {'stress_data': {'$each': item['stress_data']}}}, upsert=True)
    elif isinstance(data, dict):
        data["_id"] = str(uuid1().hex)
        result = db['prediction_results'].insert_one(data)
    else:
        raise ValueError("Invalid data format. Expected a dictionary or list of dictionaries.")
