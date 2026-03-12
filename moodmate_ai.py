import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Music dictionary
music = {
    "Happy": ["happy_song1.mp3","happy_song2.mp3"],
    "Sad": ["calm_song1.mp3","calm_song2.mp3"],
    "Angry": ["relax_song1.mp3"],
    "Surprise": ["party_song1.mp3"],
    "Neutral": ["lofi_song1.mp3"]
}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

   

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = face/255.0
        face = np.reshape(face,(1,48,48,3))

        prediction = model.predict(face)
        emotion = emotions[np.argmax(prediction)]

        # Recommend song
        if emotion in music:
            song = random.choice(music[emotion])
            print("Emotion detected:", emotion)
            print("Recommended song:", song)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("MoodMate AI",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()