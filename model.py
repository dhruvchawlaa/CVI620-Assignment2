from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from data_preprocessing import INPUT_SHAPE

def build_model():
    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE),
        Conv2D(24, (5, 5), activation='relu', strides=(2, 2)),
        Conv2D(36, (5, 5), activation='relu', strides=(2, 2)),
        Conv2D(48, (5, 5), activation='relu', strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1)),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.summary()
    return model