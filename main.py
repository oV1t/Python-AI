from src.generate_data import generate_data
from src.train_model import train_and_save_model
from src.evaluate import evaluate_model
from src.plot import plot_prediction

df = generate_data()

model = train_and_save_model(df)

mae, mse, y_pred = evaluate_model(model, df)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

plot_prediction(df, y_pred)
