import random

music = {
    "Happy": ["happy_song1.mp3","happy_song2.mp3"],
    "Sad": ["calm_song1.mp3","calm_song2.mp3"],
    "Angry": ["relax_song1.mp3"],
    "Surprise": ["party_song1.mp3"],
    "Neutral": ["lofi_song1.mp3"]
}

def recommend(emotion):
    if emotion in music:
        song = random.choice(music[emotion])
        print("Emotion detected:", emotion)
        print("Recommended song:", song)
    else:
        print("Emotion not found")

# run test
recommend("Happy")
