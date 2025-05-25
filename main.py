from preprocessing import load_and_preprocess_data
from model import train_model
from evaluate import evaluate_model
from visualize import plot_metrics

def main():
    path = 'data/internship_candidates_cefr_final.csv'
    X, y = load_and_preprocess_data(path)
    
    model, X_test, y_test, y_pred = train_model(X, y)
    
    evaluate_model(y_test, y_pred)
    plot_metrics(y_test, y_pred)

if __name__ == '__main__':
    main()
