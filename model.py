from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(layers.Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1)) 
    
    model.compile(optimizer='adam', loss='mse')
    
    return model
