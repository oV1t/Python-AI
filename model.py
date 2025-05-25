from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Створення та тренування логістичної регресії
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Прогнозування
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred
