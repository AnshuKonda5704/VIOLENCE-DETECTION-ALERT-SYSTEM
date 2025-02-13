# # app.py
# import numpy as np

# # Monkey-patch np.sctypes for compatibility with imgaug in NumPy 2.x
# # This directly injects the attribute into numpy's __dict__
# np.__dict__['sctypes'] = {
#     'int': [np.int8, np.int16, np.int32, np.int64],
#     'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
#     'float': [np.float16, np.float32, np.float64],
#     'complex': [np.complex64, np.complex128]
# }

# from flask import Flask, request, jsonify 
# from flask_cors import CORS
# import os
# import cv2
# import numpy as np
# import math
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import DepthwiseConv2D
# import imgaug.augmenters as iaa
# import imgaug as ia

# # ---------------------- Global Variables ----------------------
# IMG_SIZE = 128
# ColorChannels = 3

# # ---------------------- Video to Frames Function ----------------------
# def video_to_frames(video):
#     """
#     Extracts frames from the video file.
#     Processes only every 7th frame, applies augmentations, converts BGR to RGB,
#     and resizes the frame to (IMG_SIZE x IMG_SIZE).
#     """
#     vidcap = cv2.VideoCapture(video)
#     count = 0
#     ImageFrames = []
#     while vidcap.isOpened():
#         # Get the current frame number
#         ID = vidcap.get(1)
#         success, image = vidcap.read()
#         if success:
#             # Process only every 7th frame to avoid duplication
#             if (ID % 7 == 0):
#                 # Define augmentation operations
#                 flip = iaa.Fliplr(1.0)
#                 zoom = iaa.Affine(scale=1.3)
#                 random_brightness = iaa.Multiply((1, 1.3))
#                 rotate = iaa.Affine(rotate=(-25, 25))
                
#                 # Apply augmentations sequentially
#                 image_aug = flip(image=image)
#                 image_aug = random_brightness(image=image_aug)
#                 image_aug = zoom(image=image_aug)
#                 image_aug = rotate(image=image_aug)
                
#                 # Convert from BGR to RGB and resize
#                 rgb_img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
#                 resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
#                 ImageFrames.append(resized)
#             count += 1
#         else:
#             break
#     vidcap.release()
#     return ImageFrames

# # ---------------------- Model Loading ----------------------
# def custom_depthwise_conv2d(*args, **kwargs):
#     """
#     Custom DepthwiseConv2D to ignore the 'groups' argument.
#     This is required if your saved model was built with an extra 'groups' parameter.
#     """
#     if 'groups' in kwargs:
#         kwargs.pop('groups')
#     return DepthwiseConv2D(*args, **kwargs)

# def load_your_model(model_path):
#     """
#     Loads the pre-trained model from the given path using the custom DepthwiseConv2D.
#     """
#     model = load_model(model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
#     return model

# # ---------------------- Flask App Setup ----------------------
# app = Flask(__name__)
# CORS(app)  # Allow requests from your front end (e.g., running on localhost:3000)

# # Load the model once on startup
# model = load_your_model(r"C:\Users\anshu\Desktop\violence_detection\Violence-Alert-System\Violence Detection\modelnew.h5")
# print("Pre-trained model loaded successfully!")

# # ---------------------- API Endpoint ----------------------
# @app.route('/api/detect', methods=['POST'])
# def detect():
#     # Check if a video file is provided in the request
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided.'}), 400

#     video_file = request.files['video']
#     # Save the uploaded video temporarily
#     temp_video_path = 'temp_video.mp4'
#     video_file.save(temp_video_path)
    
#     # Extract frames using the video_to_frames function
#     frames = video_to_frames(temp_video_path)
#     if not frames:
#         os.remove(temp_video_path)
#         return jsonify({'error': 'No frames extracted from video.'}), 400
    
#     # Preprocess frames: convert pixel values to float32 and normalize to [0,1]
#     processed_frames = [frame.astype('float32') / 255.0 for frame in frames]
#     processed_frames = np.array(processed_frames)  # Shape: (num_frames, IMG_SIZE, IMG_SIZE, 3)
    
#     # Run predictions using the loaded model
#     predictions = model.predict(processed_frames)
#     # For binary classification using a sigmoid output, predictions are probabilities
#     preds = predictions > 0.5
#     n_violence = int(np.sum(preds))
#     n_total = processed_frames.shape[0]
    
#     # Remove the temporary video file
#     os.remove(temp_video_path)
    
#     # Return the detection results as a JSON response
#     return jsonify({
#         'message': f"Violence detected in {n_violence} out of {n_total} frames.",
#         'violence_frames': n_violence,
#         'nonviolence_frames': n_total - n_violence
#     })

# # ---------------------- Run the Flask App ----------------------
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


#------------------------------------------------------------------------------------------------------------
# app.py

import numpy as np

# Monkey-patch np.sctypes for compatibility with imgaug in NumPy 2.x
np.__dict__['sctypes'] = {
    'int': [np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
    'float': [np.float16, np.float32, np.float64],
    'complex': [np.complex64, np.complex128]
}

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import imgaug.augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# ---------------------- Global Variables ----------------------
IMG_SIZE = 128
ColorChannels = 3

# Global dictionary to accumulate daily counts of violence-detected videos.
daily_counts = {}

# ---------------------- Video to Frames Function ----------------------
def video_to_frames(video):
    """
    Extracts frames from the video file.
    Processes only every 7th frame, applies augmentations, converts BGR to RGB,
    and resizes the frame to (IMG_SIZE x IMG_SIZE).
    """
    vidcap = cv2.VideoCapture(video)
    count = 0
    ImageFrames = []
    while vidcap.isOpened():
        # Get the current frame number
        ID = vidcap.get(1)
        success, image = vidcap.read()
        if success:
            # Process only every 7th frame to avoid duplication
            if (ID % 7 == 0):
                # Define augmentation operations
                flip = iaa.Fliplr(1.0)
                zoom = iaa.Affine(scale=1.3)
                random_brightness = iaa.Multiply((1, 1.3))
                rotate = iaa.Affine(rotate=(-25, 25))
                
                # Apply augmentations sequentially
                image_aug = flip(image=image)
                image_aug = random_brightness(image=image_aug)
                image_aug = zoom(image=image_aug)
                image_aug = rotate(image=image_aug)
                
                # Convert from BGR to RGB and resize
                rgb_img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
                ImageFrames.append(resized)
            count += 1
        else:
            break
    vidcap.release()
    return ImageFrames

# ---------------------- Model Loading ----------------------
def custom_depthwise_conv2d(*args, **kwargs):
    """
    Custom DepthwiseConv2D to ignore the 'groups' argument.
    This is required if your saved model was built with an extra 'groups' parameter.
    """
    if 'groups' in kwargs:
        kwargs.pop('groups')
    return DepthwiseConv2D(*args, **kwargs)

def load_your_model(model_path):
    """
    Loads the pre-trained model from the given path using the custom DepthwiseConv2D.
    """
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
    return model

# ---------------------- Flask App Setup ----------------------
app = Flask(__name__)
CORS(app)  # Allow requests from your front end (e.g., running on localhost:3000)

# Load the model once on startup
model = load_your_model(r"C:\Users\anshu\Desktop\violence_detection\Violence-Alert-System\Violence Detection\modelnew.h5")
print("Pre-trained model loaded successfully!")

# ---------------------- API Endpoint ----------------------
@app.route('/api/detect', methods=['POST'])
def detect():
    # Record the upload time
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if a video file is provided in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    # Save the uploaded video temporarily
    temp_video_path = 'temp_video.mp4'
    video_file.save(temp_video_path)
    
    # Record processing start time
    start_time = datetime.now()
    
    # Extract frames using the video_to_frames function
    frames = video_to_frames(temp_video_path)
    if not frames:
        os.remove(temp_video_path)
        return jsonify({'error': 'No frames extracted from video.'}), 400
    
    # Preprocess frames: convert pixel values to float32 and normalize to [0,1]
    processed_frames = [frame.astype('float32') / 255.0 for frame in frames]
    processed_frames = np.array(processed_frames)  # Shape: (num_frames, IMG_SIZE, IMG_SIZE, 3)
    
    # Run predictions using the loaded model
    predictions = model.predict(processed_frames)
    preds = predictions > 0.5  # Binary classification threshold
    n_violence = int(np.sum(preds))
    n_total = processed_frames.shape[0]
    
    # Remove the temporary video file
    os.remove(temp_video_path)
    
    # Record processing end time and calculate duration
    end_time = datetime.now()
    processing_duration = (end_time - start_time).total_seconds()
    detection_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Update the daily report: if any violent frames are detected, count this video as a violent detection.
    detection_date = end_time.strftime("%Y-%m-%d")
    if n_violence > 0:
        daily_counts[detection_date] = daily_counts.get(detection_date, 0) + 1

    # Generate a graph representing the number of violent videos detected per day
    # Sort the dates and corresponding counts
    sorted_dates = sorted(daily_counts.keys())
    counts = [daily_counts[date] for date in sorted_dates]
    
    fig, ax = plt.subplots()
    ax.plot(sorted_dates, counts, marker="o", color="#6200ee")
    ax.set_title("Violence Videos Detected Per Day", color="#ffffff")
    ax.set_xlabel("Date", color="#e0e0e0")
    ax.set_ylabel("Count", color="#e0e0e0")
    ax.tick_params(axis='x', colors='#e0e0e0', rotation=45)
    ax.tick_params(axis='y', colors='#e0e0e0')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor='#121212', edgecolor='none')
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # Build the JSON response with a detailed report
    response = {
        'upload_time': upload_time,
        'detection_time': detection_time,
        'processing_duration': processing_duration,
        'message': f"Violence detected in {n_violence} out of {n_total} frames.",
        'violence_frames': n_violence,
        'nonviolence_frames': n_total - n_violence,
        'daily_report': daily_counts,
        'graph': graph_base64
    }
    return jsonify(response)

# ---------------------- Run the Flask App ----------------------
if __name__ == '__main__':
    app.run(port=5000, debug=True)
