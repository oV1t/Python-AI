import numpy as np
import pandas as pd

def generate_data(x_min=-20, x_max=20, num_points=500):
    x = np.linspace(x_min, x_max, num_points)
    y = np.sin(x) + 0.1 * x ** 2
    df = pd.DataFrame({'x': x, 'y': y})
    return df
