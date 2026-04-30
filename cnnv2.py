import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

IMG_SIZE   = 96      # bigger input — dust needs texture detail
BATCH_SIZE = 32
L2         = 2e-4

# --------------------------
# Generators — add stronger augmentation since dust is subtle
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,          # dust looks same upside down
    rotation_range=30,
    zoom_range=0.2,
    brightness_range=[0.6, 1.4], # simulate different lighting conditions
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    channel_shift_range=30.0,    # simulate color temperature shifts
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
# Squeeze-and-Excitation block — lets model focus on dusty channels
def se_block(x, ratio=4):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])   # recalibrate channel weights

# Depthwise separable block with SE attention
def ds_block(x, filters, stride=1, use_se=False):
    residual = x
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same',
                                use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    if use_se:
        x = se_block(x)              # attention on channels
    if stride == 1 and residual.shape[-1] == filters:
        x = layers.Add()([x, residual])
    x = layers.ReLU(6.)(x)
    return x

# --------------------------
# Build model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Stem
x = layers.Conv2D(16, 3, strides=2, padding='same',
                  use_bias=False,
                  kernel_regularizer=regularizers.l2(L2))(inputs)   # 48x48x16
x = layers.BatchNormalization()(x)
x = layers.ReLU(6.)(x)

# Backbone
x = ds_block(x, 24,  stride=2, use_se=False)   # 24x24x24
x = ds_block(x, 24,  stride=1, use_se=False)
x = ds_block(x, 40,  stride=2, use_se=True)    # 12x12x40  ← SE here
x = ds_block(x, 40,  stride=1, use_se=True)
x = ds_block(x, 64,  stride=2, use_se=True)    # 6x6x64   ← SE here
x = ds_block(x, 64,  stride=1, use_se=False)

# Head
x = layers.Conv2D(128, 1, use_bias=False,
                  kernel_regularizer=regularizers.l2(L2))(x)        # 6x6x128
x = layers.BatchNormalization()(x)
x = layers.ReLU(6.)(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

total_params = model.count_params()
print(f"Parameters : {total_params:,}")
print(f"Float32    : {total_params * 4 / 1024:.1f} KB")
print(f"INT8 quant : {total_params / 1024:.1f} KB")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --------------------------
# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_esp32_v2.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=80,
    callbacks=callbacks
)