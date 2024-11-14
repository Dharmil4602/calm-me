import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

scaler = StandardScaler()

# Function to calculate the accuracy of the model
def accuracy_score_pred(model, x_label, y_label):
  X_prediction = model.predict(x_label)
  data_accuracy = accuracy_score(y_label, X_prediction)
  return data_accuracy

# Function to generate Classification Report
def generate_classification_report(model, x_label, y_label):
  X_prediction = model.predict(x_label)
  report = classification_report(X_prediction, y_label)
  return print(f"Report for {model} is: ", report)

def get_classification_report_metrics(model, X_test, y_true, model_name):
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
# Function to predict score on user input data
def user_input_accuracy_pred(model, age, gender, bmi, hrv, hrmax):
  input_data = (age, gender, bmi, hrv, hrmax)
  input_data_as_nparray = np.asarray(input_data)
  input_data_reshaped = input_data_as_nparray.reshape(1, -1)
  std_data = scaler.transform(input_data_reshaped)
  prediction = model.predict(std_data)

  if prediction[0] == 0:
    return 'Mild Stress'
  elif prediction[0] == 1:
    return 'No Stress'
  elif prediction[0] == 2:
    return 'Stress'
  else:
    return 'Unknown'
  # if prediction[0] == 0:
  #   return 'Mild Stressed'
  # elif prediction[0] == 1:
  #   return 'No Stress'
  # elif prediction[0] == 2:
  #   return 'Stressed'
  # else:
  #   return 'Unknown'
