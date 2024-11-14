import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from models.models import svm_model, logistic_regression_model, random_forest_model, lstm_model, load_model

def load_and_preprocess_data(data_path):
    dataset = pd.read_csv(data_path)
    hrv_data = dataset[['Age', 'Gender', 'SDNN_P']]
    hrv_dataframe = pd.DataFrame(hrv_data)
    
    # Assigning Stress Labels based on the research paper
    def assign_stress_label(age, gender, sdnn):
        if age >= 18 and age <= 24:
            if sdnn > 50:
                return "Stress"
            elif 35 < sdnn <= 50:
                return "Mild Stress"
            else:
                return "No Stress"
        elif age >= 25 and age <= 34:
            if sdnn > 60:
                return "Stress"
            elif 45 < sdnn <= 60:
                return "Mild Stress"
            else:
                return "No Stress"
        elif age >= 35 and age <= 44:
            if sdnn > 40:
                return "Stress"
            elif 30 < sdnn <= 40:
                return "Mild Stress"
            else:
                return "No Stress"
        # if age >= 18 and age <= 24:
        #     if sdnn > 50:
        #         return "No Stress"
        #     elif 35 < sdnn <= 50:
        #         return "Mild Stress"
        #     else:
        #         return "Stress"
        # elif age >= 25 and age <= 34:
        #     if sdnn > 60:
        #         return "No Stress"
        #     elif 45 < sdnn <= 60:
        #         return "Mild Stress"
        #     else:
        #         return "Stress"
        # elif age >= 35 and age <= 44:
        #     if sdnn > 40:
        #         return "No Stress"
        #     elif 30 < sdnn <= 40:
        #         return "Mild Stress"
        #     else:
        #         return "Stress"
    
    hrv_dataframe['Stress_Level'] = hrv_dataframe.apply(lambda row: assign_stress_label(row['Age'], row['Gender'], row['SDNN_P']), axis=1)
    
    label_encoder = LabelEncoder()
    hrv_dataframe['Stress_Level'] = label_encoder.fit_transform(hrv_data['Stress_Level'])
    
    standard_scaler = StandardScaler()
    standardized_hrv_data = standard_scaler.fit_transform(hrv_dataframe)
    
    dataframe = pd.DataFrame(standardized_hrv_data, columns=['Age', 'Gender', 'SDNN_P', 'Stress_Level'])
    
    X = dataframe[['Age', 'Gender', 'SDNN_P']]
    Y = dataframe['Stress_Level']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
    
    training_data =  X_train, Y_train
    
    svm_model(training_data)
    logistic_regression_model(training_data)
    random_forest_model(training_data)
    lstm_model(training_data)