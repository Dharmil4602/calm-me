import numpy as np
import pandas as pd
from models.models import load_model2
from flask import Flask, request, jsonify
from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
from database.db import Connection
from uuid import uuid1
from pydantic import BaseModel
from typing import Union
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# app = Flask(__name__)
app = FastAPI()
if __name__ == "__main__":
    app.run(debug=True)
    
db = Connection(os.getenv("MONGODB_DB"))
scaler = StandardScaler()

# class Data(BaseModel):
#     email: str
#     age: int | None = None
#     gender: str | None = None
#     healthData: list

class HealthData(BaseModel):
    value: float
    dateFrom: str
    dateTo: str

# Define main schema
class Data(BaseModel):
    email: str
    age: int | None = None
    gender: str | None = None
    healthData: list[HealthData]

# FastAPI
@app.post("/predict")
async def predict(data: Union[list[Data], Data]):
    print("Dataa before: ", data)
    if isinstance(data, Data):
        # data = [dict(data)]
        data = [data]
        print("Dataa in if: ", data)

    print("Dataa: ", data)
    data_dicts = [item.model_dump() for item in data]
    dataFromFunction = loadHrvInModel(data_dicts)
    
    register_user(dataFromFunction)
    return {"message": dataFromFunction}

# Flask API
# @app.route("/predict", methods=['POST'])
# def predict():
#     data = request.get_json()
#     # Checking if the data is a list or a single object
#     if not isinstance(data, list):
#         data = [data]        
#     dataFromFunction = loadHrvInModel(data)
    
#     register_user(dataFromFunction)
#     return {"message": dataFromFunction}

# Working loadHrvInModel function
# def loadHrvInModel(data):
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
            
            model = load_model2(model_name)
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

# LoadHrvModel for Fast API
def loadHrvInModel(data: list[dict]):
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
            
            model_name = 'feed'
            
            model = load_model2(model_name)
            prediction = model.predict(input_data_scaled)
            
            # stress_mapping = {0: "Chronic Stress", 1: "Low Stress", 2: "Mild Stress"}
            # prediction_result = stress_mapping.get(int(prediction[0]), "Unknown")
            # label = 0 if prediction_result == "Chronic Stress" else 1 if prediction_result == "Low Stress" else 2
            
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    # If prediction is a 2D array, take the first element
                    pred_value = prediction[0][0]
                else:
                    # If prediction is a 1D array, take the first element
                    pred_value = prediction[0]
            else:
                # If prediction is already a scalar
                pred_value = prediction
                
            # Convert to integer for mapping
            pred_value = int(np.round(pred_value))
            
            stress_mapping = {0: "Chronic Stress", 1: "Low Stress", 2: "Mild Stress"}
            prediction_result = stress_mapping.get(pred_value, "Unknown")
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
