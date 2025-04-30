import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize = (8,6))

    plt.scatter(y_test, y_pred, alpha=0.6)

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
        color='red', linestyle='--')
    plt.title("Справжня vs Прогнозована ціна")
    plt.xlabel("Справжня ціна")
    plt.ylabel("Прогнозована ціна")
    plt.grid(True)
    plt.show()