import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import base64
from datetime import datetime
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess

# Initialize server
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# Speed parameters
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED
prev_steering = 0.0  # Initialize previous steering value

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering, speed_limit
    
    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return

    try:
        # Extract data from simulator
        steering_angle = float(data['steering_angle'])
        throttle = float(data['throttle'])
        speed = float(data['speed'])
        
        # Process image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image_array = np.asarray(image)
        processed_image = preprocess(image_array)
        processed_image = np.array([processed_image])  # Add batch dimension
        
        # Predict steering angle
        predicted_steering = float(model.predict(processed_image, verbose=0)[0])
        
        # Apply smoothing to steering
        smoothed_steering = 0.7 * prev_steering + 0.3 * predicted_steering
        prev_steering = smoothed_steering
        
        # Dynamic speed limit adjustment
        if speed > speed_limit:
            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED
        
        # Calculate throttle (more conservative for sharp turns)
        throttle = 1.0 - (smoothed_steering**2) - (speed/speed_limit)**2
        throttle = np.clip(throttle, 0.1, 0.5)  # Keep between 0.1 and 0.5
        
        print(f"Steering: {smoothed_steering:.3f}, Throttle: {throttle:.3f}, Speed: {speed:.2f}")
        
        # Send control commands
        send_control(smoothed_steering, throttle)
        
        # Save images if recording
        if args.image_folder:
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_path = os.path.join(args.image_folder, f"{timestamp}.jpg")
            image.save(image_path)
            
    except Exception as e:
        print(f"Error: {str(e)[:200]}")
        send_control(0.0, 0.0)  # Stop the car if error occurs

@sio.on('connect')
def connect(sid, environ):
    print("\n=== Simulator Connected ===")
    send_control(0.0, 0.0)  # Initialize with zero controls

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Self-driving car simulation')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model, compile=False)
    
    # Create image folder if specified
    if args.image_folder:
        print(f"Creating image folder at {args.image_folder}")
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        print("*** RECORDING THIS RUN ***")
    else:
        print("*** NOT RECORDING THIS RUN ***")

    # Start server
    app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except KeyboardInterrupt:
        print("\nShutting down server...")