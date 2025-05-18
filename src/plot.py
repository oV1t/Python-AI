import matplotlib.pyplot as plt

def plot_prediction(df, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(df['x'], df['y'], label='Реальна функція', color='blue')
    plt.plot(df['x'], y_pred, label='Прогноз моделі', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Передбачення функції f(x) = sin(x) + 0.1x²')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
