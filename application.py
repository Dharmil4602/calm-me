import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from models.models import load_model
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler
from database.db import Connection
from uuid import uuid1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# app = FastAPI()
app = Flask(__name__)
db = Connection(os.getenv("MONGODB_DB")) # Database Connection
scaler = StandardScaler()
# scaler = joblib.load('standard_scaler.save')
class StressPredictionInput(BaseModel):
    age: int
    gender: int
    hrv: float
    bmi: float
    hrmax: int

@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

# # Fast API
# @app.post("/predict/")
# def predict(data: StressPredictionInput, model_name = 'logistic_regression'):
#     input_data = np.asarray([[data.age, data.gender, data.hrv, data.hrmax, data.bmi]])
#     print("..................", input_data)
    
#     dataset = pd.read_csv('./data/hrv_dataset.csv')
#     hrv_dataframe = pd.DataFrame(dataset, columns=['Age', 'Gender', 'BMI', 'SDNN_P', 'Hrmax'])    
    
#     scaler.fit(hrv_dataframe)
    
#     input_data1 = scaler.transform(input_data)
#     print('21111111111111', input_data1)
    
#     model = load_model(model_name)
#     prediction = model.predict(input_data1)
#     print("ppppppppppp", prediction)
    
#     print(data.hrv)

#     stress_mapping = {0: "Mild Stress", 1: "No Stress", 2: "Stress"}
#     return {"model": model_name, "prediction": stress_mapping.get(int(prediction), "Unknown")}


# Flask API
@app.route("/predict/", methods=['POST'])
def predict():
    data = request.get_json()
    
    print("Data: ", data)
    
    age = data['age']
    gender = data['gender']
    hrv = data['hrv']
    hrmax = data['hrmax']
    bmi = data['bmi']
    
    input_data = np.asarray([[age, gender, hrv, hrmax, bmi]])
    
    print("Input Data:", input_data)
    
    dataset = pd.read_csv('./data/hrv_dataset.csv')
    hrv_dataframe = pd.DataFrame(dataset, columns=['Age', 'Gender', 'BMI', 'SDNN_P', 'Hrmax'])
    
    scaler.fit(hrv_dataframe)
    input_data_scaled = scaler.transform(input_data)
    
    model_name = 'svm'
    
    model = load_model(model_name)
    prediction = model.predict(input_data_scaled)
    
    stress_mapping = {0: "Mild Stress", 1: "No Stress", 2: "Stress"}
    prediction_result = stress_mapping.get(int(prediction[0]), "Unknown")
    register_user({
        "age": age,
        "gender": gender,
        "hrv": hrv,
        "hrmax": hrmax,
        "bmi": bmi,
        "model": model_name,
        "prediction": prediction_result
    })
    return {"model": model_name, "prediction": prediction_result}


def register_user(data=None):
    if data is None:
        data = dict(request.json)
    
    print("User Data from input: ", data)
    
    if data['gender'] == 1:
        data.update({"gender": "male"})
    elif data['gender'] == 0:
        data.update({"gender": "female"})
    
    _id = str(uuid1().hex)
    data.update({"_id": _id})
    
    result = db['user-data'].insert_one(data)
    if not result.inserted_id:
        return {"message": "Failed to insert user data"}

    return {"message": "User data inserted successfully"}