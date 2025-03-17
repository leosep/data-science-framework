import pandas as pd

def collect_data(config):
    if config["source"] == "csv":
        return pd.read_csv(config["file_path"])
    else:
        raise ValueError("Unsupported data source")
