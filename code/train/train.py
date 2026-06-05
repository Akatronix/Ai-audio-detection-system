import pandas as pd
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# --- 1. CONFIGURATION ---
DATA_ROOT = "UrbanSound8K"
METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")

SR = 16000
N_MELS = 64
DURATION = 3 
INPUT_SHAPE = (64, 94, 1)

# --- 2. FEATURE EXTRACTION ---
def extract_features(fold, filename):
    # Use absolute path to avoid ambiguity
    file_path = os.path.abspath(os.path.join(AUDIO_DIR, f"fold{fold}", filename))
    
    if not os.path.exists(file_path):
        return None
        
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        # Standardize length to exactly 3 seconds
        if len(audio) < SR * DURATION:
            audio = np.pad(audio, (0, SR * DURATION - len(audio)))
        else:
            audio = audio[:SR * DURATION]
            
        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel[..., np.newaxis]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# --- 3. DATA LOADING ---
df = pd.read_csv(METADATA_PATH)

def get_dataset(fold_list):
    X, y = [], []
    print(f"Loading Folds: {fold_list}...")
    subset = df[df['fold'].isin(fold_list)]
    
    # LIMIT DATA FOR INITIAL TEST: remove '.head(100)' to use all files
    for _, row in subset.iterrows():
        feat = extract_features(row['fold'], row['slice_file_name'])
        if feat is not None:
            X.append(feat)
            y.append(row['classID'])
            
    if len(X) == 0:
        return np.array([]), np.array([])
        
    return np.array(X, dtype='float32'), np.array(y, dtype='int32')

# --- 4. MODEL ARCHITECTURE ---
def build_proposed_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Proposed Separable Conv for Edge Efficiency
        layers.SeparableConv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 5. TRAINING EXECUTION (3 ITERATIONS) ---
# Each iteration uses a different combination of folds for Train/Val/Test
iterations = [
    {'train': [1,2,3,4,5,6], 'val': [7,8], 'test': [9,10]},
    {'train': [3,4,5,6,7,8], 'val': [9,10], 'test': [1,2]},
    {'train': [5,6,7,8,9,10], 'val': [1,2], 'test': [3,4]}
]

for i, config in enumerate(iterations):
    print(f"\n{'='*30}\nITERATION {i+1}\n{'='*30}")
    
    X_train, y_train = get_dataset(config['train'])
    X_val, y_val = get_dataset(config['val'])
    X_test, y_test = get_dataset(config['test'])
    
    # CRITICAL CHECK: Stop if data failed to load
    if X_val.size == 0 or X_train.size == 0:
        print(f"Error: Dataset for Iteration {i+1} is empty. Check your folder paths.")
        continue

    model = build_proposed_model()
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=15, 
        batch_size=32
    )
    
    # --- 6. PLOTTING & LOGGING ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Iteration {i+1} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Iteration {i+1} Loss')
    plt.legend()
    
    plt.savefig(f'iteration_{i+1}_results.png')
    print(f"Iteration {i+1} graphs saved.")

    # Final Evaluation
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"\nFinal Report for Iteration {i+1}:")
    print(classification_report(y_test, y_pred))
    
    # Save Model for Pi 4
    model.save(f'proposed_model_v{i+1}.h5')