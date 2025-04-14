import pandas as pd
import numpy as np
import cv2
import random
import os
from data_augmentation import augment_data

def fix_path(path):
    path = str(path)
    path = path.replace('\\', '/')  
    tail = path.split('/')[-1]
    img_path = os.path.join('Driving Log Data', 'IMG', tail)  
    return img_path

def load_data(csv_path='Driving Log Data/driving_log.csv'):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(csv_path, names=columns)
    
    data['center'] = data['center'].apply(fix_path) 

    # Ensure steering column is numeric, any errors to NaN
    data['steering'] = pd.to_numeric(data['steering'], errors='coerce')

    # Drop rows where steering value is NaN 
    data = data.dropna(subset=['steering'])
    
    print(f'Total Images Loaded: {len(data)}')
    return data

def preProcessing(img):
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize to 200x66 pixels
    img = img / 255  # Normalize the image
    return img

def load_img(path):
    if not os.path.exists(path):
        print(f"Error: Image file doesn't exist: {path}")
        return np.zeros((66, 200, 3))
    
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return np.zeros((66, 200, 3))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = preProcessing(img)
    return img

def batch_generator(data, batch_size=32, training=True):
    while True:
        data = data.sample(frac=1)  
        
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i:i + batch_size]
            
            images = []
            steerings = []
            
            for _, row in batch_data.iterrows():
                img = load_img(row['center'])
                steering = float(row['steering'])
                
                if training and random.random() > 0.5:
                    img, steering = augment_data(img, steering)  
                
                images.append(img)
                steerings.append(steering)
            
            X = np.array(images)
            y = np.array(steerings)
            
            yield X, y
