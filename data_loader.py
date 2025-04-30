# data_loader.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    num_features = ['year', 'engine_volume', 'mileage', 'horsepower']
    X_num = df[num_features]

    cat_features = ['brand', 'model']

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_cat_array = encoder.fit_transform(df[cat_features])
    cat_cols = encoder.get_feature_names_out(cat_features)
    X_cat = pd.DataFrame(X_cat_array, columns=cat_cols, index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    y = df['price']
    return X, y, encoder
