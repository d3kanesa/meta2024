from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict

def calcAngle(distA, distB):
    magnitudeA = pow(distA[0] ** 2 + distA[1] ** 2 + distA[2] ** 2, 0.5)
    magnitudeB = pow(distB[0] ** 2 + distB[1] ** 2 + distB[2] ** 2, 0.5)
    dot = distA[0] * distB[0] + distA[1] * distB[1] + distA[2] * distB[2]
    return math.acos(dot / (magnitudeA * magnitudeB)) * 180 / math.pi

def calcVector(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]

def midpoint(a, b):
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Add debug text
            cv2.putText(frame, 'Tracking', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in multipart response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
def hand_keypoints_view(request):
    # if request.headers.get('accept') == 'text/event-stream':
    #     return StreamingHttpResponse(
    #         generate_frames(),
    #         content_type='multipart/x-mixed-replace; boundary=frame'
    #     )
    return render(request, 'hand_keypoints.html')

def home_view(request):
    return render(request, 'home.html')