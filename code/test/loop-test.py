import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf  # On Windows we use the full TF we already installed
import time

# --- CONFIGURATION ---
# Use the best model you generated (Iteration 2 was your best)
# MAKE SURE YOU CONVERTED IT TO .tflite FIRST!
MODEL_PATH = "../../model/urban_audio_edge.tflite" 
SR = 16000
DURATION = 3 
N_MELS = 64
# UrbanSound8K Labels
LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
    "Drilling", "Engine Idling", "Gun Shot", "Siren", "Street Music", "Jackhammer"
]

# --- LOAD TFLITE MODEL ---
print(f"Loading model: {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio_data):
    # Padding/Clipping to exactly 3s
    if len(audio_data) < SR * DURATION:
        audio_data = np.pad(audio_data, (0, SR * DURATION - len(audio_data)))
    else:
        audio_data = audio_data[:SR * DURATION]
    
    # Feature Extraction (Matching your training code)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Add Batch and Channel dimensions: (1, 64, 94, 1)
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

print("\n[LIVE TEST] Speak into the mic or play a sound...")
print("The system will record every 3 seconds.")

try:
    while True:
        # 1. Capture Audio
        print("\n--- Listening ---")
        recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        sd.wait() # Wait until recording is finished
        audio_data = recording.flatten()
        
        # 2. Preprocess
        input_data = preprocess_audio(audio_data)
        
        # 3. Predict
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Show Results
        prediction_idx = np.argmax(output_data)
        confidence = output_data[0][prediction_idx]
        
        if confidence > 0.35: # Sensitivity threshold
            print(f"RESULT: {LABELS[prediction_idx]} ({confidence*100:.1f}%)")
        else:
            print("RESULT: Ambient/Uncertain")

except KeyboardInterrupt:
    print("\nTest stopped by user.")