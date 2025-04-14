from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()

    # Normalization layer (scale to [-0.5, 0.5] assuming input is [0, 1])
    model.add(layers.Lambda(lambda x: (x - 0.5), input_shape=(66, 200, 3)))
    
    # Convolutional layers
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1)) 
    
    model.compile(optimizer='adam', loss='mse')
    
    return model