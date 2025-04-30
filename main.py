# main.py
from data_loader import load_data
from model import train_and_evaluate
from plotter import plot_actual_vs_predicted

def main():
    X, y, encoder = load_data("data/cars_plus.csv")
    model, X_test, y_test, y_pred, r2, mape = train_and_evaluate(X, y)
    
    print(f"R^2: {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")
    
    plot_actual_vs_predicted(y_test, y_pred)

if __name__ == "__main__":
    main()
