from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

def train_model(df):
    X = df[['Quantity', 'Price']]
    y = df['TotalAmount']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'model/model.pkl')
    return model

def predict_total_amount(quantity, price, model_path='model/model.pkl'):
    model = joblib.load(model_path)
    prediction = model.predict([[quantity, price]])
    return prediction[0]
