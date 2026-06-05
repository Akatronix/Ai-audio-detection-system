import tensorflow as tf

# 1. Load your H5 model
# If your filename is different, change it here (e.g., 'proposed_model_v2.h5')
h5_model_path = '../model/proposed_model_v2.h5'

print(f"Loading {h5_model_path}...")
model = tf.keras.models.load_model(h5_model_path)

# 2. Initialize the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Convert the model
print("Converting model to TFLite format...")
tflite_model = converter.convert()

# 5. Save the TFLite file
tflite_filename = 'urban_audio_edge.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully created: {tflite_filename}")