import numpy as np
import librosa
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import time

# --- CONFIGURATION ---
MODEL_PATH = "urban_audio_edge.tflite"
SR = 16000
DURATION = 3  # seconds
N_MELS = 64
LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
    "Drilling", "Engine Idling", "Gun Shot", "Siren", "Street Music"
]

# --- LOAD TFLITE MODEL ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(audio_data):
    """Convert raw audio to the exact Mel-spec format the model expects."""
    # Ensure duration is exactly 3s
    if len(audio_data) < SR * DURATION:
        audio_data = np.pad(audio_data, (0, SR * DURATION - len(audio_data)))
    else:
        audio_data = audio_data[:SR * DURATION]
    
    # Extract Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Reshape for model: (Batch, Height, Width, Channels) -> (1, 64, 94, 1)
    log_mel = log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
    return log_mel

print("--- Audio System Active ---")
print("Listening for urban sounds... (Press Ctrl+C to stop)")

try:
    while True:
        # 1. Record 3 seconds of audio
        print("\nRecording...")
        recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        sd.wait()
        audio_data = recording.flatten()
        
        # 2. Process
        input_data = preprocess_audio(audio_data)
        
        # 3. Inference
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 4. Results
        end_time = time.time()
        prediction = np.argmax(output_data)
        confidence = output_data[0][prediction]
        
        # Only print if confidence is above a threshold
        if confidence > 0.40:
            print(f"Detected: {LABELS[prediction]} ({confidence*100:.1f}%)")
            print(f"Inference Time: {(end_time - start_time)*1000:.2f}ms")
        else:
            print("Background Noise / Unrecognized")

except KeyboardInterrupt:
    print("\nStopping system...")