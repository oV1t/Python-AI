from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_nn_model(input_dim, hidden_layers=[64, 32]):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_nn(model, X_train, y_train):
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=300,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    return model
