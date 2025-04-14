import pandas as pd
import numpy as np
import cv2
import random
import os

def fix_path(path):
    path = str(path)
    path = path.replace('\\', '/')
    tail = path.split('/')[-1]
    img_path = os.path.join('Driving Log Data', 'IMG', tail)
    return img_path

def augment_data(img, steering):
    # Flipping (50% chance)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
        steering = -steering    # Reverse steering angle
    # Brightness adjustment (50% chance)
    if np.random.rand() < 0.5:
        img = img * 255.0
        img = cv2.convertScaleAbs(img, beta=np.random.uniform(-0.2, 0.2) * 255)
        img = img / 255.0
    # Zooming (50% chance)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)  # Zoom in/out by 80%-120%
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img[start_h:start_h + h, start_w:start_w + w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT, value=0)
    # Panning (50% chance)
    if np.random.rand() < 0.5:
        max_shift = 10  # Max pixel shift
        dx = np.random.randint(-max_shift, max_shift)
        dy = np.random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=0)
    # Rotation (50% chance)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-5, 5)  # Rotate between -5 and 5 degrees
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return img, steering

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
        return None
    
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return None
    
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
                if img is None:
                    continue  # Skip if image fails to load
                steering = float(row['steering'])
                
                if training and random.random() > 0.5:
                    img, steering = augment_data(img, steering)  
                
                images.append(img)
                steerings.append(steering)
            
            X = np.array(images)
            y = np.array(steerings)
            
            yield X, y