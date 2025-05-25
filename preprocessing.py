import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_and_preprocess_data(path: str):
    # Завантаження даних
    data = pd.read_csv(path)

    # Видалення пропущених значень
    data.dropna(inplace=True)

    # Кодування EnglishLevel з урахуванням порядку
    english_levels_order = [
        'Elementary', 'Pre-Intermediate', 'Intermediate',
        'Upper-Intermediate', 'Advanced'
    ]
    encoder = OrdinalEncoder(categories=[english_levels_order])
    data[['EnglishLevel']] = encoder.fit_transform(data[['EnglishLevel']])

    # Розділення на X та y
    X = data.drop('Accepted', axis=1)
    y = data['Accepted']

    return X, y
