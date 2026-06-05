# import pandas as pd
# import numpy as np
# import librosa
# import os
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models
# from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# # --- 1. CONFIGURATION ---
# DATA_ROOT = "UrbanSound8K"
# METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "UrbanSound8K.csv")
# AUDIO_DIR = os.path.join(DATA_ROOT, "audio")

# SR = 16000
# N_MELS = 64
# DURATION = 3 
# INPUT_SHAPE = (64, 94, 1)

# # --- 2. FEATURE EXTRACTION ---
# def extract_features(fold, filename):
#     file_path = os.path.abspath(os.path.join(AUDIO_DIR, f"fold{fold}", filename))
#     if not os.path.exists(file_path):
#         return None
#     try:
#         audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
#         if len(audio) < SR * DURATION:
#             audio = np.pad(audio, (0, SR * DURATION - len(audio)))
#         else:
#             audio = audio[:SR * DURATION]
#         mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
#         log_mel = librosa.power_to_db(mel, ref=np.max)
#         return log_mel[..., np.newaxis]
#     except Exception as e:
#         print(f"Error loading {filename}: {e}")
#         return None

# # --- 3. DATA LOADING ---
# df = pd.read_csv(METADATA_PATH)

# def get_dataset(fold_list):
#     X, y = [], []
#     print(f"Loading Folds: {fold_list}...")
#     subset = df[df['fold'].isin(fold_list)]
    
#     for _, row in subset.iterrows():
#         feat = extract_features(row['fold'], row['slice_file_name'])
#         if feat is not None:
#             X.append(feat)
#             y.append(row['classID'])
            
#     if len(X) == 0:
#         return np.array([]), np.array([])
#     return np.array(X, dtype='float32'), np.array(y, dtype='int32')

# # --- 4. MODEL ARCHITECTURE ---
# def build_proposed_model():
#     model = models.Sequential([
#         layers.Input(shape=INPUT_SHAPE),
#         layers.Conv2D(32, (3,3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2,2)),
#         layers.SeparableConv2D(64, (3,3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2,2)),
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # --- 5. EVALUATION HELPER ---
# def calculate_metrics(model, X, y, iteration, split_name, folds):
#     y_pred_probs = model.predict(X)
#     y_pred = np.argmax(y_pred_probs, axis=1)
    
#     # Calculate weighted metrics
#     precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
#     acc = accuracy_score(y, y_pred)
    
#     return {
#         'Iteration': iteration,
#         'Split': split_name,
#         'Folds': str(folds),
#         'Accuracy': round(acc, 4),
#         'Precision': round(precision, 4),
#         'Recall': round(recall, 4),
#         'F1-Score': round(f1, 4)
#     }

# # --- 6. TRAINING EXECUTION ---
# iterations = [
#     {'train': [1,2,3,4,5,6], 'val': [7,8], 'test': [9,10]},
#     {'train': [3,4,5,6,7,8], 'val': [9,10], 'test': [1,2]},
#     {'train': [5,6,7,8,9,10], 'val': [1,2], 'test': [3,4]}
# ]

# all_results = []

# for i, config in enumerate(iterations):
#     print(f"\n{'='*30}\nITERATION {i+1}\n{'='*30}")
    
#     X_train, y_train = get_dataset(config['train'])
#     X_val, y_val = get_dataset(config['val'])
#     X_test, y_test = get_dataset(config['test'])
    
#     if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
#         print(f"Skipping Iteration {i+1} due to missing data.")
#         continue

#     model = build_proposed_model()
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32, verbose=1)
    
#     # Collect metrics for all sets
#     all_results.append(calculate_metrics(model, X_train, y_train, i+1, 'Training', config['train']))
#     all_results.append(calculate_metrics(model, X_val, y_val, i+1, 'Validation', config['val']))
#     all_results.append(calculate_metrics(model, X_test, y_test, i+1, 'Testing', config['test']))
    
#     # Save model
#     model.save(f'proposed_model_v{i+1}.h5')

# # --- 7. FINAL TABULATION ---
# results_df = pd.DataFrame(all_results)
# results_df.to_csv("final_evaluation_results.csv", index=False)

# print("\n" + "="*50)
# print("FINAL CONSOLIDATED RESULTS")
# print("="*50)
# print(results_df.to_string(index=False))


import pandas as pd
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# --- 1. CONFIGURATION (Path Auto-Detection) ---
# Detects path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Moves up two levels from code/train to geminiAudio root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "UrbanSound8K")
METADATA_PATH = os.path.join(DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")

SR = 16000
N_MELS = 64
DURATION = 3 
INPUT_SHAPE = (64, 94, 1)

# Debug: Print paths to console for verification
print(f"Project Root detected at: {PROJECT_ROOT}")
if not os.path.exists(METADATA_PATH):
    print(f"ERROR: Metadata not found at {METADATA_PATH}")
    print("Please ensure UrbanSound8K folder is in the project root.")

# --- 2. FEATURE EXTRACTION ---
def extract_features(fold, filename):
    file_path = os.path.abspath(os.path.join(AUDIO_DIR, f"fold{fold}", filename))
    if not os.path.exists(file_path):
        return None
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
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
# Load metadata once at the start
try:
    df = pd.read_csv(METADATA_PATH)
except Exception as e:
    print(f"Failed to read CSV: {e}")
    exit()

def get_dataset(fold_list):
    X, y = [], []
    print(f"Loading Folds: {fold_list}...")
    subset = df[df['fold'].isin(fold_list)]
    
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
        
        # Proposed Separable Conv for Edge Efficiency (Pi 4 Optimized)
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

# --- 5. EVALUATION HELPER ---
def calculate_metrics(model, X, y, iteration, split_name, folds):
    print(f"Calculating metrics for {split_name} set...")
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y, y_pred)
    
    return {
        'Iteration': iteration,
        'Split': split_name,
        'Folds': str(folds),
        'Accuracy': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

# --- 6. TRAINING EXECUTION ---
iterations = [
    {'train': [1,2,3,4,5,6], 'val': [7,8], 'test': [9,10]},
    {'train': [3,4,5,6,7,8], 'val': [9,10], 'test': [1,2]},
    {'train': [5,6,7,8,9,10], 'val': [1,2], 'test': [3,4]}
]

all_results = []

for i, config in enumerate(iterations):
    print(f"\n{'='*40}\nSTARTING ITERATION {i+1}\n{'='*40}")
    
    X_train, y_train = get_dataset(config['train'])
    X_val, y_val = get_dataset(config['val'])
    X_test, y_test = get_dataset(config['test'])
    
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print(f"Skipping Iteration {i+1} due to missing data.")
        continue

    model = build_proposed_model()
    
    # Training
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=15, 
        batch_size=32, 
        verbose=1
    )
    
    # Collect metrics for all stages (Requested by supervisor)
    all_results.append(calculate_metrics(model, X_train, y_train, i+1, 'Training', config['train']))
    all_results.append(calculate_metrics(model, X_val, y_val, i+1, 'Validation', config['val']))
    all_results.append(calculate_metrics(model, X_test, y_test, i+1, 'Testing', config['test']))
    
    # Save model and plot
    model.save(f'proposed_model_v{i+1}.h5')
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title(f'Iter {i+1} Acc')
    plt.subplot(1,2,2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title(f'Iter {i+1} Loss')
    plt.savefig(f'iteration_{i+1}_history.png')
    plt.close()

# --- 7. FINAL TABULATION ---
results_df = pd.DataFrame(all_results)

# Export results to CSV for your report
results_df.to_csv("final_evaluation_results.csv", index=False)

print("\n" + "="*60)
print("FINAL CONSOLIDATED RESULTS TABLE")
print("="*60)
print(results_df.to_string(index=False))
print("\nResults also saved to 'final_evaluation_results.csv'")