from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_poly(X_train, y_train, degree=4):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return poly, model
