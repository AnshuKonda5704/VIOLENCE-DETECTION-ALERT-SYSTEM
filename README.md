# Violence Alert System- FINAL YEAR MAJOR PROJECT

A violence detector using a `MobileNetV2` pretrained model for violence detection and a `Faster RCNN Inception V2 COCO model` for human detection, implemented in Python. This project includes a Flask backend API and a React frontend for video upload and report display.

## ARCHITECTURE DIAGRAM
![Screenshot 2025-02-13 191230](https://github.com/user-attachments/assets/58b0a1c8-b222-4c63-99ff-d01adff6fe6d)

## STEP 1 (Human Detection) 

- **Description:**  
  A real-time human detector using the `Faster RCNN Inception V2 COCO model` to compare three pretrained models for speed and accuracy.

## STEP 2  (Violence Detection) 

- **Description:**  
  A real-time violence detector using the `MobileNetV2` pretrained model. The system processes video frames using OpenCV, overlays detection results, and returns a detailed report including processing times and frame counts. The training file for the MobileNetV2 model, along with testing files and videos, are included.

## FRONTEND

- **Technology:** React  
- **Description:**  
  A user-friendly interface that lets users upload video files and view detailed detection reports (including processing time, detection time, and a graph showing daily violent detections).

## BACKEND

- **Description:**  
  A REST API that accepts video uploads, processes them using the violence detection model, and returns a JSON report with:
  - Upload time, detection time, and processing duration.
  - Number of violent and non-violent frames.
  - A graph (base64-encoded) representing the number of violent videos detected per day.

## How to Run (Windows)

### Backend

1. **Open Command Prompt in the backend folder:**
   ```bat
   cd C:\Users\anshu\Desktop\violence_detection\backend
2. Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install flask flask-cors opencv-python numpy tensorflow imgaug matplotlib

4. Place your model file (modelnew.h5) in the backend folder (or update the path in app.py accordingly).
Run the Flask server:

python app.py
The server will run on http://localhost:5000.


Frontend
1. Open Command Prompt in the frontend folder:

cd C:\Users\anshu\Desktop\violence_detection\violence-detection-frontend

2.Install dependencies:
npm install
npm install axios

3. Start the React app:

npm start
The app will open in your browser at http://localhost:3000.


Testing the System
Upload a Video:
Use the React interface to select and upload a video file.

View the Report:
The backend processes the video and returns a detailed report including:

Upload time, detection time, and processing duration.
Number of violent and non-violent frames.
A graph image showing the daily count of violent detections.
 Results  screenshots:



![Screenshot (155)](https://github.com/user-attachments/assets/54634dbf-5d66-464c-8f27-4b67ba1ebe42)

![Screenshot (156)](https://github.com/user-attachments/assets/93774bea-c58e-4e65-bc02-44cbe054e13f)


![Screenshot (157)](https://github.com/user-attachments/assets/bf257ed7-9470-46e4-a15b-7edd96a8c8fe)
This README provides a brief project description and Windows-specific instructions, and result screenshots. 
