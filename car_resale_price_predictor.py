import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib 
import os 
import re 
import numpy as np

model_filename = 'car_resale_model.pkl'
encoders_filename = 'encoders.pkl'

data = pd.read_csv('car_resale_prices.csv')

def clean_kms_driven(kms_str):
    if pd.isna(kms_str):
        return 0
    return float(re.sub(r'[^\d.]', '', str(kms_str)))

def clean_resale_price(price_str):
    if pd.isna(price_str):
        return 0
    price_str = re.sub(r'[^\d.]', '', price_str)  
    return float(price_str)

data['kms_driven'] = data['kms_driven'].apply(clean_kms_driven)
data['resale_price'] = data['resale_price'].apply(clean_resale_price)

encoders = {}

def apply_label_encoding(column_name):
    encoder = LabelEncoder()
    data[column_name] = encoder.fit_transform(data[column_name])
    encoders[column_name] = encoder

apply_label_encoding('full_name')
apply_label_encoding('transmission_type')
apply_label_encoding('fuel_type')
apply_label_encoding('owner_type')

X = data[['full_name', 'transmission_type', 'fuel_type', 'owner_type', 'kms_driven']]
y = data['resale_price']

def train_and_save_model():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    joblib.dump(model, model_filename)
    joblib.dump(encoders, encoders_filename)
    print(f"Model and encoders saved to disk as '{model_filename}' and '{encoders_filename}'.")

def load_model():
    if os.path.exists(model_filename) and os.path.exists(encoders_filename):
    
        model = joblib.load(model_filename)
        encoders = joblib.load(encoders_filename)
        print(f"Loaded model and encoders from '{model_filename}' and '{encoders_filename}'.")
        return model, encoders
    else:
        print(f"Model not found. Training a new one.")
        train_and_save_model()
        return load_model() 

def predict_resale_value(model, encoders, full_name, transmission_type, fuel_type, owner_type, kms_driven):
    
    try:
        full_name_encoded = encoders['full_name'].transform([full_name])[0]
    except ValueError:
        print(f"Warning: '{full_name}' is a new model not seen during training. Encoding as 0.")
        full_name_encoded = 0
    
    transmission_type_encoded = encoders['transmission_type'].transform([transmission_type])[0]
    fuel_type_encoded = encoders['fuel_type'].transform([fuel_type])[0]
    owner_type_encoded = encoders['owner_type'].transform([owner_type])[0]

    
    input_data = pd.DataFrame([[full_name_encoded, transmission_type_encoded, fuel_type_encoded, owner_type_encoded, kms_driven]], 
                              columns=['full_name', 'transmission_type', 'fuel_type', 'owner_type', 'kms_driven'])

    predicted_value = model.predict(input_data)[0]
    return predicted_value


def get_user_input():
    print("Please provide the following details about your car:")
    full_name = input("Full Model Name (e.g., 2015 Maruti Swift Dzire VXI): ")
    transmission_type = input("Transmission Type (Manual/Automatic): ")
    fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ")
    owner_type = input("Owner (First/Second): ")
    kms_driven = float(input("Kilometers Driven: ").replace(',', ''))  

    return full_name, transmission_type, fuel_type, owner_type, kms_driven

model, encoders = load_model()

full_name, transmission_type, fuel_type, owner_type, kms_driven = get_user_input()

resale_value = predict_resale_value(model, encoders, full_name, transmission_type, fuel_type, owner_type, kms_driven)
print(f"The predicted resale value for your car is: {resale_value}")
