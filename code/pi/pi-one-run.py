import numpy as np
import librosa
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import time
import os
import RPi.GPIO as GPIO
from scipy import stats

# --- CONFIGURATION ---
MODEL_PATH = "urban_audio_edge.tflite"
SR = 16000
DURATION = 3 
N_MELS = 64
NUM_SAMPLES = 3  # Takes 3 samples to average the result
BUTTON_PIN = 17

LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
    "Drilling", "Engine Idling", "Gun Shot", "Siren", "Street Music", "Jackhammer"
]

# --- SETUP HARDWARE & MODEL ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def speak(text):
    """Uses espeak to say the result out loud."""
    print(f"TTS: {text}")
    os.system(f'espeak "{text}"')

def preprocess_audio(audio_data):
    if len(audio_data) < SR * DURATION:
        audio_data = np.pad(audio_data, (0, SR * DURATION - len(audio_data)))
    else:
        audio_data = audio_data[:SR * DURATION]
    mel = librosa.feature.melspectrogram(y=audio_data, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

def run_detection_cycle():
    """Captures multiple samples, averages results, and speaks."""
    detected_classes = []
    inference_times = []
    
    speak("Starting detection")
    
    for i in range(NUM_SAMPLES):
        print(f"Recording Sample {i+1}...")
        recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        sd.wait()
        audio_data = recording.flatten()
        
        # Inference
        input_data = preprocess_audio(audio_data)
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_times.append((time.time() - start) * 1000)
        
        detected_classes.append(np.argmax(output_data))

    # Calculate Consensus
    final_idx = stats.mode(detected_classes, keepdims=True).mode[0]
    final_label = LABELS[final_idx]
    avg_latency = np.mean(inference_times)
    
    result_text = f"Detected {final_label}. Average latency {int(avg_latency)} milliseconds."
    speak(result_text)

print("System Ready. Press the button on GPIO 17 to start detection...")

try:
    while True:
        # Check if button is pressed (Active Low)
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == False:
            run_detection_cycle()
            time.sleep(1) # Debounce
        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("System Shutdown")