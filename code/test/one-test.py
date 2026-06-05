import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import time
from scipy import stats # For calculating the most frequent result

# --- CONFIGURATION ---
MODEL_PATH = "../../model/urban_audio_edge.tflite" 
SR = 16000
DURATION = 3 
N_MELS = 64
NUM_SAMPLES = 5  # We will take 5 samples to get a solid average
LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
    "Drilling", "Engine Idling", "Gun Shot", "Siren", "Street Music", "Jackhammer"
]

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio_data):
    if len(audio_data) < SR * DURATION:
        audio_data = np.pad(audio_data, (0, SR * DURATION - len(audio_data)))
    else:
        audio_data = audio_data[:SR * DURATION]
    mel = librosa.feature.melspectrogram(y=audio_data, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

# --- EXECUTION ---
inference_times = []
detected_classes = []

print(f"\n[SYSTEM START] Testing {NUM_SAMPLES} samples for average performance...")

for i in range(NUM_SAMPLES):
    print(f"Sample {i+1}/{NUM_SAMPLES}: Recording 3 seconds...")
    
    # 1. Capture
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    audio_data = recording.flatten()
    
    # 2. Preprocess & Time the Inference
    input_data = preprocess_audio(audio_data)
    
    start_time = time.time() # Start Clock
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time() # Stop Clock
    
    # 3. Store Data
    inference_times.append((end_time - start_time) * 1000) # Convert to ms
    prediction_idx = np.argmax(output_data)
    detected_classes.append(prediction_idx)

# --- FINAL OUTPUT CALCULATION ---
avg_time = np.mean(inference_times)
# Find the most common result in our samples
final_prediction_idx = stats.mode(detected_classes, keepdims=True).mode[0]
final_label = LABELS[final_prediction_idx]

print("\n" + "="*40)
print("FINAL EXPERIMENTAL RESULT")
print("="*40)
print(f"Detected Sound  : {final_label}")
print(f"Avg Latency     : {avg_time:.2f} ms")
print(f"Total Test Time : {NUM_SAMPLES * DURATION} seconds")
print("="*40)
print("System stopping...")