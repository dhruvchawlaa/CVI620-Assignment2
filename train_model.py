import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger
from data_preprocessing import INPUT_SHAPE, batch_generator
from model import build_model

def load_data(csv_path='Driving Log Data/driving_log.csv', data_dir='.'):
    data_df = pd.read_csv(csv_path, header=None, skiprows=1)
    
    X = data_df[[0, 1, 2]].values  # First 3 columns are center/left/right image paths
    y = data_df[3].astype(float).values  # 4th column is steering angle
    
    return train_test_split(X, y, test_size=0.2, random_state=0)

def train_model(model, X_train, X_valid, y_train, y_valid, data_dir='.'):
    csv_logger = CSVLogger('training_log.csv')
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='auto')
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit(
        batch_generator(data_dir, X_train, y_train, 40, True),
        steps_per_epoch=len(X_train) // 40,  # Scale with data size
        epochs=20,  # Increase from 10 to 20
        validation_data=batch_generator(data_dir, X_valid, y_valid, 40, False),
        validation_steps=len(X_valid) // 40,
        callbacks=[checkpoint,csv_logger],
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