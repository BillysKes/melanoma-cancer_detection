


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomTranslation, RandomCrop, Resizing, Rescaling
)
from tensorflow.keras.utils import image_dataset_from_directory


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

zoom_factor = 0.2
crop_height = int(224 * (1 - zoom_factor))
crop_width = int(224 * (1 - zoom_factor))

train_datagen = tf.keras.Sequential([
    Rescaling(1./255),  # Rescales pixel values to [0, 1]
    RandomFlip("horizontal"),  # Random horizontal flip
    RandomRotation(factor=0.2),  # Rotates images randomly up to 20 degrees
    RandomTranslation(height_factor=0.2, width_factor=0.2),  # Shifts images vertically and horizontally up to 20%
    RandomCrop(height=crop_height, width=crop_width),  # Randomly crop the image
    Resizing(height=224, width=224)  # Resizes back to the original
])

test_datagen = tf.keras.Sequential([Rescaling(1./255)])

train_dataset = image_dataset_from_directory(
    "Melanoma Cancer Image Dataset/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,  # Batch size for training
    label_mode="binary"  # Binary classification (Melanoma/Non-Melanoma)
)

test_dataset = image_dataset_from_directory(
    'Melanoma Cancer Image Dataset/test',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary')

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

train_steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy() // BATCH_SIZE
val_steps_per_epoch = tf.data.experimental.cardinality(test_dataset).numpy() // BATCH_SIZE
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=10,
    validation_data=test_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_dataset, steps=val_steps_per_epoch)
print('Test accuracy:', test_acc)

plt.figure(figsize=(12, 4))
# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.show()