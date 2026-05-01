import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os

# --------------------------
# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
L2 = 2e-4
EPOCHS = 100
LEARNING_RATE = 3e-3
LABEL_SMOOTHING = 0.1

# --------------------------
# Data generators – milder augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'cropped_dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    'cropped_dataset/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# --------------------------
# Squeeze-and-Excitation block
def se_block(x, ratio=4):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

# Depthwise separable block – fixed regularizer
def ds_block(x, filters, stride=1, use_se=False):
    residual = x
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                               depthwise_regularizer=regularizers.l2(L2))(x)   # <- FIXED
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    if use_se:
        x = se_block(x)
    if stride == 1 and residual.shape[-1] == filters:
        x = layers.Add()([x, residual])
    x = layers.ReLU(6.)(x)
    return x

# --------------------------
# Build the model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Stem
x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False,
                  kernel_regularizer=regularizers.l2(L2))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU(6.)(x)   # 64x64x16

# Backbone
x = ds_block(x, 24, stride=2, use_se=False)   # 64x64 -> 32x32
x = ds_block(x, 24, stride=1, use_se=False)
x = ds_block(x, 40, stride=2, use_se=True)    # 32x32 -> 16x16
x = ds_block(x, 40, stride=1, use_se=True)
x = ds_block(x, 64, stride=2, use_se=True)    # 16x16 -> 8x8
x = ds_block(x, 64, stride=1, use_se=False)

# Head
x = layers.Conv2D(128, 1, use_bias=False, kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU(6.)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Parameter count
total_params = model.count_params()
print(f"\nParameters: {total_params:,}")
print(f"FP32 model size: {total_params * 4 / 1024:.1f} KB")
print(f"INT8 quantised approx: {total_params / 1024:.1f} KB")
model.summary()

# Compile with label smoothing
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss,
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_tiny_cnn_esp32.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
