import numpy as np

def generate_data(n_samples=300, seed=42):
    np.random.seed(seed)
    hours = np.linspace(0, 23.99, n_samples)
    duration = (
        30 + 10 * np.sin((hours - 6) * np.pi / 12) +
        5 * np.random.randn(n_samples)
    )
    return hours.reshape(-1, 1), duration
