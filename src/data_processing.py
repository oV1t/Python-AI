import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['TotalAmount'] = df['Quantity'] * df['Price']
    return df
