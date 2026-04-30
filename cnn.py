import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# ESP32 safe config
IMG_SIZE   = 48      # small input — critical for RAM
BATCH_SIZE = 32
L2         = 1e-4

# --------------------------
# Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
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

# --------------------------
# Tiny model designed for <200KB
def build_esp32_model(input_size=48):
    inputs = layers.Input(shape=(input_size, input_size, 3))

    # Block 1
    x = layers.Conv2D(8, 3, strides=2, padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(inputs)   # 24x24x8
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)   # ReLU6 — better for quantization

    # Block 2 — depthwise separable
    x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(16, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    # Block 3 — stride 2
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(32, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)        # 12x12x32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    # Block 4 — stride 2
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(48, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)        # 6x6x48
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    x = layers.GlobalAveragePooling2D()(x)   # 48 values only
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

model = build_esp32_model()
model.summary()

# Count params
total_params = model.count_params()
print(f"\nTotal parameters : {total_params:,}")
print(f"Estimated size   : {total_params * 4 / 1024:.1f} KB (float32)")
print(f"After INT8 quant : {total_params / 1024:.1f} KB")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_esp32_model.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=60,
    callbacks=callbacks
)