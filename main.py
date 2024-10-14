import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.models import load_model

app = FastAPI()
class StressPredictionInput(BaseModel):
    age: int
    gender: int
    hrv: float
    bmi: float
    hrmax: int

@app.get('/')
def hello():
    return "Hello World!"

@app.post("/predict/")
def predict(data: StressPredictionInput, model_name = 'svm'):
    # dataFromUser = ()
    input_data = np.asarray([[data.age, data.gender, data.hrv, data.hrmax, data.bmi]])
    
    model = load_model(model_name)
    prediction = model.predict(input_data)
    # if model_name == "svm":
    #     model = svm_model()
    #     prediction = model.predict(input_data)
    # elif model_name == "logistic_regression":
    #     model = logistic_regression_model()
    #     prediction = model.predict(input_data)
    # elif model_name == "random_forest":
    #     model = random_forest_model()
    #     prediction = model.predict(input_data)

    stress_mapping = {0: "Mild Stress", 1: "No Stress", 2: "Stress"}
    return {"model": model_name, "prediction": stress_mapping.get(int(prediction[0]), "Unknown")}