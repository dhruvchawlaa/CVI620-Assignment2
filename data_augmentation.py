import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(image, steering_angle):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    image = np.expand_dims(image, axis=0)
    aug_iter = datagen.flow(image, batch_size=1)
    augmented_image = next(aug_iter)[0]
    
    # If the image is flipped, the steering angle should be reversed
    if np.random.rand() < 0.5:  
        augmented_image = np.fliplr(augmented_image)
        steering_angle = -steering_angle

    return augmented_image, steering_angle