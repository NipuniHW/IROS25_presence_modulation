# MIT License
# 
# Copyright (c) [2024] Modulate Presence
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#Import relevant libraries
import pyaudio
import numpy as np
import librosa
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import wave
import time
import re
from transformers import pipeline
import joblib
from tensorflow.keras.models import load_model
from connection import Connection
import qi
from playsound import playsound
from dotenv import load_dotenv
from tensorflow.python.client import device_lib
from openai import OpenAI

# os.system("aplay mic.wav")  # to test the recorded audio


# Connect Pepper robot
pepper = Connection()
session = pepper.connect(IP Address, Port)

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts_service = session.service("ALTextToSpeech")

os.chdir(Add the home directory)

# Initialize OpenAI client
client = OpenAI(api_key=(Add the API key))

# Load pre-trained model for sentiment analysis
nlp = pipeline("sentiment-analysis")

# Load the CNN model for ambient sound detection
model = load_model(Add the path to the ambien sound detection model - emergency_model.h5)
input_shape = model.input_shape[1:] 

# Load the trained Naive Bayes model for final state classification
nb_model = joblib.load(Add the path to naive bayes model - final_NB_model.joblib)

# Function to generate the prompt for Pepper
def generate_gpt_prompt(final_label, transcription):
    if final_label == "Alarmed":
        messages = 'You are Pepper, an interactive friendly agent who will inform on an emegency situation. You will inform people to evacuate immediately. You use short sentences. You use maximum of 2 sentences.'
    elif final_label == "Alert":
        messages = 'You are Pepper, an interactive friendly agent who wants to verify whether it is an emergency situation or not. You use short sentences. You use maximum of 2 sentences.'
    elif final_label == "Social":
        messages = 'You are Pepper, an interactive friendly agent who is chatty and loves to engage in casual conversations. Do not say who you are except for the name. Do not say "as an AI". You use short sentences. You use maximum of 2 sentences.'
    elif final_label == "Passive":
        messages = 'You are Pepper, an interactive friendly agent who will stay low and does not speak much.'
    elif final_label == "Disengaged":
        messages = 'Use 0 words.'
      
    # Call the OpenAI API to generate the appropriate response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": messages}, 
            {"role": "user", "content": transcription}
        ]
    )
    
    # Extract the GPT-generated response
    generated_prompt = response.choices[0].message.content
    print(f"GPT-generated prompt: {generated_prompt}")
    
    return generated_prompt

# Play a state relevant behaviour and speak
def animation(final_label, prompt_text):
    behavior_mng_service.stopAllBehaviors()
    
    # Play specific behavior for the identified context
    behavior_mng_service.startBehavior("presence_module/" + final_label)
    
    # Pepper says the prompt
    tts_service.say(prompt_text)

# Function to extract MFCCs from real-time audio for ambient sound classification
def preprocess_audio(audio, sr, n_mfcc=40, n_fft=2048, hop_length=512, fixed_length=200):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Pad or truncate the MFCC features to fixed_length
    if mfccs.shape[1] < fixed_length:
        pad_width = fixed_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :fixed_length]

    return mfccs

# Function to classify ambient sound
def classify_real_time_audio(model, input_shape, sr=16000):
    p = pyaudio.PyAudio()
    chunk_size = 1024
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sr,
                    input=True,
                    input_device_index=13,
                    frames_per_buffer=chunk_size)

    frames = []
    for i in range(0, int(sr / chunk_size * 3)):  # 3 seconds
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.float32))

    audio = np.concatenate(frames, axis=0)
    mfccs = preprocess_audio(audio, sr=sr, fixed_length=input_shape[1])
    mfccs = mfccs.reshape(1, *input_shape)

    prediction = model.predict(mfccs)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    class_labels = ['Alarmed', 'Social', 'Disengaged']

    # Save recorded audio as mic.wav
    with wave.open("mic.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))

    if (class_labels[predicted_class]=='Alarmed'):
        sentiment_confidence = max(confidence, 0.6)
        return predicted_class, sentiment_confidence, class_labels[predicted_class]
    elif (class_labels[predicted_class]=='Social'):
        sentiment_confidence = max(confidence*0.5, 0.3)
        return predicted_class, sentiment_confidence, class_labels[predicted_class]
    elif (class_labels[predicted_class]=='Disengaged'):
        sentiment_confidence = confidence*0.2
        return predicted_class, sentiment_confidence, class_labels[predicted_class]

# Function to process speech-to-text transcription and analyze sentiment
def process_speech_to_text_and_sentiment():
    file_path = "mic.wav"
    
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    transcription_text = transcription.text
    print(f"Transcription: {transcription_text}")

    # Filter transcription to only allow English text (removes non-ASCII characters, emojis, etc.)
    transcription_text = re.sub(r'[^a-zA-Z\s]', '', transcription_text).strip()

    if len(transcription_text) == 0:
        return 2, 0, 0, "Disengaged", transcription_text

    # Sentiment analysis on the transcribed text
    sentiment_result = nlp(transcription_text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    alarmed_keywords = ['emergency', 'help', 'accident', 'fire', 'danger']
    found_keywords = any(word in transcription_text.lower() for word in alarmed_keywords)

    if (sentiment_label == 'NEGATIVE' and sentiment_score > 0.7) and found_keywords:
        return 0, max(sentiment_score, 0.7), 1, "Alarmed", transcription_text

    social_keywords = ['hi', 'hello', 'hey', 'how are you']
    identified = any(word in transcription_text.lower() for word in social_keywords)

    if identified and (sentiment_label == 'NEGATIVE'):
        return 1, sentiment_score, 0.4, "Social", transcription_text
    if not identified and (sentiment_label == 'NEGATIVE'):
        return 1, sentiment_score, 0.35, "Social", transcription_text
    if identified and (sentiment_label == 'POSITIVE'):
        return 1, (1 - sentiment_score), 0.4, "Social", transcription_text
    if not identified and (sentiment_label == 'POSITIVE'):
        return 1, (1 - sentiment_score), 0.35, "Social", transcription_text

# Real-time listener and processor
def listen_and_process():
    sr = 16000 
    while True:
        # Step 1: Record and classify ambient sound
        ambient_class, ambient_conf, ambient_label = classify_real_time_audio(model, input_shape, sr=sr)

        # Step 2: Process speech-to-text and sentiment analysis
        speech_class, sentiment_conf, keyword_conf, speech_label, transcription_text = process_speech_to_text_and_sentiment()

        # Step 3: Combine results using Naive Bayes
        context_label, final_label = classify_context(ambient_conf, keyword_conf, sentiment_conf)

        # Display the results
        print(f"Ambient: {ambient_label} (Conf: {ambient_conf:.2f}), Speech: {speech_label} (Keyword Conf: {keyword_conf:.2f}) (Sentiment Conf: {sentiment_conf: .2f})")
        print(f"Final Context: {final_label}\n")

        # Step 4: Generate and execute behavior based on context
        prompt_text = generate_gpt_prompt(final_label, transcription_text)
        animation(final_label, prompt_text)

        # Press 'q' to quit Pepper
        user_input = input("Press 'q' to quit or any other key to continue: ").lower()
        if user_input == 'q':
            print("Stopping Pepper's behaviors...")
            behavior_mng_service.stopAllBehaviors()  
            print("Quitting...")
            break

# Function to combine the two models using Naive Bayes for context classification
def classify_context(ambient_confidence, keyword_confidence, sentiment_confidence):
    X = np.array([[ambient_confidence, keyword_confidence, sentiment_confidence]])

    # Get the predicted class and the corresponding probability for each class
    combined_class = nb_model.predict(X)[0]
    class_probs = nb_model.predict_proba(X)[0]
    
    class_labels = ['Alarmed', 'Social', 'Disengaged']
    context_label = class_labels[combined_class]
    
    # Print the probabilities for each class
    print(f"Naive Bayes Confidence: {dict(zip(class_labels, class_probs))}")
    final_label = ['Alarmed', 'Alert', 'Social', 'Passive', 'Disengaged']
    prob = class_probs[combined_class]
    if context_label == 'Disengaged' and 0.8 < prob < 1.1:
        final_label = 'Disengaged'
    elif context_label == 'Disengaged' and 0 < prob < 0.79:
        final_label = 'Passive'
    elif context_label == 'Social' and 0.8 < prob < 1.1:
        final_label = 'Social' #should be Social
    elif context_label == 'Social' and 0 < prob < 0.79:
        final_label = 'Passive'
    elif context_label == 'Alarmed' and 0.8 < prob < 1.1:
        final_label = 'Alarmed' #should be Alert
    elif context_label == 'Alarmed' and 0 < prob < 0.79:
        final_label = 'Alert'

    return combined_class, final_label

# Start the real-time listener
listen_and_process()
