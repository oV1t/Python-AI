from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib

def train_and_save_model(df, model_path="model/model.pkl", degree=4):
    X = df[['x']]
    y = df['y']
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    return model
