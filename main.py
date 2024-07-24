import import_ipynb
from NNetwork import CNN
import os
import tempfile
import cv2
import torch
import numpy as np
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import torch.nn.functional as F

# Define the model and setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = FastAPI()
model = CNN()  # Assuming CNN is defined in NNetwork.py
model.to(device)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'Saved_models', 'model3', 'model3.pt'), map_location=device))
model.eval()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
analyzer = SentimentIntensityAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class SentimentResponse(BaseModel):
    sentiment: str
    score: float

def most_common_string(strings):
    counter = Counter(strings)
    most_common = counter.most_common(1)[0]
    most_common_string, count = most_common
    return most_common_string, count

async def predict_image(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_image, 1.1, 4)
    if len(faces) == 0:
        return -1
    
    face_roi = None
    for (x, y, w, h) in faces:
        roi_grey = gray_image[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_grey)
        if len(facess) == 0:
            continue
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]
            break

    if face_roi is None:
        return -1

    face_roi_grey = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    final_img = cv2.resize(face_roi_grey, (48, 48))
    final_img = np.expand_dims(final_img, axis=0)
    final_img = np.expand_dims(final_img, axis=0)
    final_img = final_img / 255.0
    test_img = torch.from_numpy(final_img).to(device).float()

    with torch.no_grad():
        prediction = model(test_img)
        op = F.softmax(prediction, dim=1)

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_class = torch.argmax(op).item()

    return emotions[predicted_class]


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    try:
        with open(temp_file_path, 'wb') as f:
            f.write(await file.read())
    finally:
        temp_file.close()
    
    # Extract audio
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    try:
        with VideoFileClip(temp_file_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_audio_path)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio = recognizer.record(source)
            try:
                audio_text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return JSONResponse(content={"message": "Audio could not be understood or mute audio"}, status_code=400)
            except sr.RequestError:
                return JSONResponse(content={"message": "Error with the speech recognition service"}, status_code=500)

        sentiment_score = analyzer.polarity_scores(audio_text)
        sentiment = "Neutral"
        if sentiment_score['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment = "Negative"

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # Process video frames
    video_file = cv2.VideoCapture(temp_file_path)
    if not video_file.isOpened():
        return JSONResponse(content={"message": "Failed to open video file"}, status_code=400)
    
    resp = []
    while True:
        ret, frame = video_file.read()
        if not ret:
            break
        
        # Process each frame for emotion prediction
        prediction = await predict_image(frame)
        if prediction != -1:
            resp.append(prediction)

    video_file.release()
    os.remove(temp_file_path)
    
    if not resp:
        return JSONResponse(content={"message": "No faces detected in video"}, status_code=400)

    emo, _ = most_common_string(resp)
    
    return JSONResponse(content={"Emotion": emo, "Sentiment": sentiment, "Text": audio_text}, status_code=200)
