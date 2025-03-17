from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def build_model(data, config):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-config["train_test_split"], random_state=42)

    if config["model_type"] == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate model
    if "rmse" in config["metrics"]:
        print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False)}")
    if "r2" in config["metrics"]:
        print(f"R² Score: {r2_score(y_test, predictions)}")
