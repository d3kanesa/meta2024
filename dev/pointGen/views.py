import json
from urllib import request
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import cv2
import mediapipe as mp
import math
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse, JsonResponse
from rest_framework import status
import numpy as np
import time
from pygame import mixer
from mutagen.mp3 import MP3
from .ml import main as MLMAIN  # Import the analyze_form function from ml.py
import asyncio
from collections import defaultdict

CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/SAz9YHcvj6GT2YYXdXww"

d=defaultdict(list)

headers = {
  "Accept": "audio/wav",
  "Content-Type": "application/json",
  "xi-api-key": "sk_85e693c5e23fe7cfcf28738eb391022472d568e534e7f735"
}

def calcAngle(distA, distB):
    magnitudeA = pow(distA[0] ** 2 + distA[1] ** 2 + distA[2] ** 2, 0.5)
    magnitudeB = pow(distB[0] ** 2 + distB[1] ** 2 + distB[2] ** 2, 0.5)
    dot = distA[0] * distB[0] + distA[1] * distB[1] + distA[2] * distB[2]
    return math.acos(dot / (magnitudeA * magnitudeB)) * 180 / math.pi

def calcVector(a, b):
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]

def midpoint(a, b):
    return [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

non_face_landmarks = {
    name: idx
    for name, idx in mp_pose.PoseLandmark.__members__.items()
    if "LEFT_EYE" not in name and "RIGHT_EYE" not in name and
    "MOUTH" not in name and "NOSE" not in name and
    "EAR" not in name
}

is_processing = False  

def generate_frames():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if is_processing:
                frame = process_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

def process_frame(frame):
    global d
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for name, idx in non_face_landmarks.items():
            if name not in d:
                d[name] = {}
                d[name]["x"] = []
                d[name]["y"] = []
                d[name]["z"] = []
            d[name]["x"].append(landmarks[idx].x)
            d[name]["y"].append(landmarks[idx].y)
            d[name]["z"].append(landmarks[idx].z)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    return frame

async def genContent():
    print("Generating content")
    for i,j in d.items():
        d[i]["x"] = [sum(j["x"])/len(j["x"])]
        d[i]["y"] = [sum(j["y"])/len(j["y"])]
        d[i]["z"] = [sum(j["z"])/len(j["z"])]
        print(d[i],"\n\n\n\n")
    await MLMAIN(str(d))

async def handle_recording(request):
    global is_processing
    global d
    if request.method == 'POST':
        data = json.loads(request.body)
        action = data.get('action')
        if action == 'start_recording':
            is_processing = True
            clearD()
            return JsonResponse({'status': 'success', 'message': 'Recording started'})
        elif action == 'stop_recording':
            is_processing = False
            await genContent()
            return JsonResponse({'status': 'success', 'message': 'Recording stopped'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid action'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def clearD():
    global d
    d.clear()

def hand_keypoints_view(request):
    if request.method=="POST":
        print("Hi!")
    # if request.headers.get('accept') == 'text/event-stream':
    #     return StreamingHttpResponse(
    #         generate_frames(),
    #         content_type='multipart/x-mixed-replace; boundary=frame'
    #     )
    return render(request, 'hand_keypoints.html')

def home_view(request):
    return render(request, 'home.html')

def provideFeedback(feedback):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/SAz9YHcvj6GT2YYXdXww"

    headers = {
    "Accept": "audio/wav",
    "Content-Type": "application/json",
    "xi-api-key": "sk_85e693c5e23fe7cfcf28738eb391022472d568e534e7f735"
    }
    
    data = {
        "text": feedback,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = request.post(url, json=data, headers=headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    mixer.init()
    mixer.music.load("C:\\Users\\saaru\\Documents\\LLama Hackathon\\output.mp3")
    mixer.music.play()
    time.sleep(MP3("output.mp3").info.length)