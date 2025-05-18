from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, df):
    X = df[['x']]
    y_true = df['y']
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    return mae, mse, y_pred
