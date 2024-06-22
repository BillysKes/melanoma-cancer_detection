import os
import gc
import tensorflow as tf
from keras.src.layers import RandomBrightness
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomTranslation, RandomCrop, RandomBrightness, Resizing, Rescaling)
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import Policy


K.clear_session()

# Ensure GPU is being used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPU not available. Using CPU.")

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


print(tf.test.is_built_with_cuda())
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

zoom_factor = 0.2
crop_height = int(IMAGE_SIZE[0] * (1 - zoom_factor))
crop_width = int(IMAGE_SIZE[1] * (1 - zoom_factor))

train_datagen = tf.keras.Sequential([
    Rescaling(1./255),  # Rescales pixel values to [0, 1]
    RandomFlip(),  # Random horizontal flip
    RandomRotation(factor=0.25),  # Rotates images randomly up to 20 degrees
#    RandomBrightness(factor=0.25)  # Add random brightness adjustment
    RandomTranslation(height_factor=0.2, width_factor=0.2),  # Shifts images vertically and horizontally up to 20%
])

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

# Checking cardinality of the datasets
num_train_samples = tf.data.experimental.cardinality(train_dataset).numpy() * BATCH_SIZE
num_test_samples = tf.data.experimental.cardinality(test_dataset).numpy() * BATCH_SIZE

print("Number of train samples:", num_train_samples)
print("Number of test samples:", num_test_samples)

train_dataset = train_dataset.map(lambda x, y: (train_datagen(x), y)).repeat()
test_dataset = test_dataset.map(lambda x, y: (Rescaling(1./255)(x), y)).repeat()

# Calculate steps per epoch based on dataset size
train_steps_per_epoch = num_train_samples // BATCH_SIZE
val_steps_per_epoch = num_test_samples // BATCH_SIZE

print("Train steps per epoch:", train_steps_per_epoch)
print("Validation steps per epoch:", val_steps_per_epoch)

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-3)),
    BatchNormalization(),  # Add Batch Normalization after the first Dense layer
    Dropout(0.2),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=5e-6)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=test_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks=[early_stopping, reduce_lr]
)

tf.keras.backend.clear_session()
gc.collect()

test_loss, test_acc = model.evaluate(test_dataset, steps=val_steps_per_epoch)
print('Test accuracy:', test_acc)


tf.keras.backend.clear_session()
gc.collect()


# Making predictions on the test dataset
test_dataset = test_dataset.take(val_steps_per_epoch)  # Ensure the test dataset is not infinitely repeated
test_images, test_labels = zip(*(list(test_dataset.as_numpy_iterator())))
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int).reshape(-1)

tf.keras.backend.clear_session()
gc.collect()

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# Classification report
class_report = classification_report(test_labels, predicted_labels, target_names=['Benign', 'Malignant'])
print("Classification Report:")
print(class_report)

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
plt.savefig('training_validation_plots.png')

