import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from data_preprocessing import INPUT_SHAPE, batch_generator
from model import build_model

def load_data(csv_path='Driving Log Data/driving_log.csv', data_dir='.'):
    columns = [
        'Driving Log Data\\IMG\\Center',
        'Driving Log Data\\IMG\\Left',
        'Driving Log Data\\IMG\\Right',
        'Steering',
        'Throttle',
        'Brake',
        'Speed'
    ]
    data_df = pd.read_csv(csv_path, names=columns, skiprows=1)
    X = data_df[['Driving Log Data\\IMG\\Center', 'Driving Log Data\\IMG\\Left', 'Driving Log Data\\IMG\\Right']].values
    y = data_df['Steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid

def train_model(model, X_train, X_valid, y_train, y_valid, data_dir='.'):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='auto')
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(
        batch_generator(data_dir, X_train, y_train, 40, True),
        steps_per_epoch=len(X_train) // 40,  # Scale with data size
        epochs=20,  # Increase from 10 to 20
        validation_data=batch_generator(data_dir, X_valid, y_valid, 40, False),
        validation_steps=len(X_valid) // 40,
        callbacks=[checkpoint],
        verbose=1
    )

def main():
    data_dir = '.'
    data = load_data(data_dir=data_dir)
    model = build_model()
    train_model(model, *data, data_dir=data_dir)
    model.save('model.h5')

if __name__ == '__main__':
    main()