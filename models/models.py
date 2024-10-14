from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import os

def svm_model(X_train, Y_train):
    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(X_train, Y_train)
    
    model_path = 'models/saved/svm_model.pkl'
    joblib.dump(model, model_path)

def logistic_regression_model(X_train, Y_train):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    model_path='models/saved/logistic_model.pkl'
    joblib.dump(model, model_path)

def random_forest_model(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    
    model_path='models/saved/rf_model.pkl'
    joblib.dump(model, model_path)

def lstm_model(X_train, Y_train):
    Y_train = to_categorical(Y_train)
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=50, batch_size=32)
    
    filename = 'lstm_model.h5'
    model.save(filename)

def load_model(model_name):
    current_dir = os.path.dirname(__file__)
    if model_name == 'svm':
        model_path = os.path.join(current_dir, 'svm_model.joblib')
        model = joblib.load(model_path)
        return model
    elif model_name == 'logistic_regression':
        model_path = os.path.join(current_dir, 'logistic_model.joblib')
        model = joblib.load(model_path)
        return model
    elif model_name == 'random_forest':
        model_path = os.path.join(current_dir, 'rf_model.joblib')
        model = joblib.load(model_path)
        return model
    elif model_name == 'lstm':
        return joblib.load('lstm_model.h5')
    else:
        return None