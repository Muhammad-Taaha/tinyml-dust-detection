import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import glob
import os

# ============================
# CONFIGURATION
# ============================
MODEL_PATH = 'best_tiny_cnn_esp32.keras'   # your trained model
IMAGE_DIR = 'cropped_dataset/train'        # folder with subfolders 'clean' and 'dusty'
IMG_SIZE = 128                             # must match training input size
NUM_CALIBRATION_IMAGES = 100

# ============================
# Load the Keras model
# ============================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ============================
# Representative dataset generator (yields float32, range [0,1])
# ============================
def representative_dataset_gen():
    # Collect image paths from subfolders
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*', '*.jpg'))
    if len(image_paths) < NUM_CALIBRATION_IMAGES:
        image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    paths = image_paths[:NUM_CALIBRATION_IMAGES]
    print(f"Using {len(paths)} images for calibration.")

    for img_path in paths:
        # Load and preprocess exactly as during training
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)                # shape (128,128,3) uint8
        img_array = img_array / 255.0                      # normalise to [0,1] float32
        img_batch = np.expand_dims(img_array, axis=0)      # shape (1,128,128,3) float32
        yield [img_batch.astype(np.float32)]

# ============================
# Convert to INT8 quantized TFLite
# ============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Let the converter decide input/output types (they will be int8)
# Do NOT manually set inference_input_type / inference_output_type

tflite_model = converter.convert()

# ============================
# Save the quantized model
# ============================
output_path = 'tiny_cnn_esp32_int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

model_size_kb = len(tflite_model) / 1024
print(f"Quantized model saved to {output_path} (size: {model_size_kb:.2f} KB)")