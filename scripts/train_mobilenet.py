import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------
# Configuration
IMG_TARGET_SIZE = 224      # MobileNetV2 native size — better features
BATCH_SIZE = 32
NUM_CLASSES = 1
L2 = 1e-4

# --------------------------
# Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    'cropped_dataset/train',
    target_size=(IMG_TARGET_SIZE, IMG_TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    'cropped_dataset/val',
    target_size=(IMG_TARGET_SIZE, IMG_TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --------------------------
# Phase 1 — Freeze base, train head only
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_TARGET_SIZE, IMG_TARGET_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze all pretrained weights

inputs = layers.Input(shape=(IMG_TARGET_SIZE, IMG_TARGET_SIZE, 3))
x = base_model(inputs, training=False)  # training=False keeps BN frozen
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(L2))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

print("=" * 50)
print("PHASE 1: Training head only (base frozen)")
print("=" * 50)
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

# --------------------------
# Phase 2 — Unfreeze top layers of base for fine-tuning
print("=" * 50)
print("PHASE 2: Fine-tuning top layers of base model")
print("=" * 50)

base_model.trainable = True

# Only unfreeze the last 30 layers — earlier layers have universal features
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Much lower LR to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-8, verbose=1),
    ModelCheckpoint('best_model_finetuned.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks_phase2
)