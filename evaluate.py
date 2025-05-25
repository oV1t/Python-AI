from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_test, y_pred):
    print("Оцінка моделі:")
    print("Точність (Accuracy):", accuracy_score(y_test, y_pred))
    print("Прецизійність (Precision):", precision_score(y_test, y_pred))
    print("Повнота (Recall):", recall_score(y_test, y_pred))
    print("F1-міра:", f1_score(y_test, y_pred))
