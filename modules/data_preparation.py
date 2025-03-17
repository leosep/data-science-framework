import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, config):
    # Handle missing values
    if config["missing_values"] == "mean":
        data.fillna(data.mean(), inplace=True)
    elif config["missing_values"] == "median":
        data.fillna(data.median(), inplace=True)

    # Normalize data
    if config["normalize"]:
        scaler = MinMaxScaler()
        data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    
    return data
