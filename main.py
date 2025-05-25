import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_generator import generate_data
from model_nn import build_nn_model, train_nn
from model_poly import train_poly
from utils import time_to_hour

# --- Генерація та розбиття даних
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Нейронна мережа
nn_model = build_nn_model(input_dim=1, hidden_layers=[64, 32])
nn_model = train_nn(nn_model, X_train, y_train)
nn_preds = nn_model.predict(X_test).flatten()

# --- Поліноміальна регресія
poly, poly_model = train_poly(X_train, y_train)
X_test_poly = poly.transform(X_test)
poly_preds = poly_model.predict(X_test_poly)

# --- Метрики
print("\n--- MAE / MSE ---")
print("Neural Network:")
print("MAE:", mean_absolute_error(y_test, nn_preds))
print("MSE:", mean_squared_error(y_test, nn_preds))

print("Polynomial Regression:")
print("MAE:", mean_absolute_error(y_test, poly_preds))
print("MSE:", mean_squared_error(y_test, poly_preds))

# --- Прогноз для заданих часів
times = ["10:30", "00:00", "02:40"]
times_float = np.array([time_to_hour(t) for t in times]).reshape(-1, 1)
nn_pred_times = nn_model.predict(times_float).flatten()
poly_pred_times = poly_model.predict(poly.transform(times_float)).flatten()

print("\n--- Передбачення тривалості поїздки ---")
for i, t in enumerate(times):
    print(f"{t} -> NN: {nn_pred_times[i]:.2f} хв, Poly: {poly_pred_times[i]:.2f} хв")

# --- Візуалізація
plt.figure(figsize=(12, 6))
plt.scatter(X, y, label='Дані', alpha=0.5)
x_grid = np.linspace(0, 24, 300).reshape(-1, 1)
plt.plot(x_grid, nn_model.predict(x_grid), label='Нейронна мережа', color='blue')
plt.plot(x_grid, poly_model.predict(poly.transform(x_grid)), label='Поліноміальна регресія', color='green')
plt.xlabel('Час доби (години)')
plt.ylabel('Тривалість поїздки (хв)')
plt.title('Передбачення тривалості поїздки')
plt.legend()
plt.grid(True)
plt.show()
