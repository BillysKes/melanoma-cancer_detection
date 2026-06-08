import os
import gc
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, CenterCrop, Resizing, Rescaling)
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure GPU is being used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# 1. Setup Base Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 2. Augmentation Pipelines
train_datagen = tf.keras.Sequential([
    Resizing(height=224, width=224, interpolation="bilinear"),
    RandomFlip(),  
    RandomRotation(factor=0.25),  
    Rescaling(1./255)
])

# Test/Val data should ONLY be resized and rescaled, never flipped or rotated
test_datagen = tf.keras.Sequential([
    Resizing(height=224, width=224, interpolation="bilinear"), 
    Rescaling(1./255)
])

# 3. Proper Data Splitting (Train, Val, Test)
print("Loading Training and Validation Data...")
train_dataset = image_dataset_from_directory(
    "Melanoma Cancer Image Dataset/train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,  
    label_mode="binary"  
)

val_dataset = image_dataset_from_directory(
    "Melanoma Cancer Image Dataset/train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,  
    label_mode="binary"  
)

print("Loading Unseen Test Data...")
test_dataset = image_dataset_from_directory(
    'Melanoma Cancer Image Dataset/test',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Crucial for accurate confusion matrix alignment later
)

# Apply augmentations and scaling
train_dataset = train_dataset.map(lambda x, y: (train_datagen(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (test_datagen(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (test_datagen(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# 4. Build Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-3)),
    BatchNormalization(),  
    Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=5e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train Model (Using the isolated validation set)
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

tf.keras.backend.clear_session()
gc.collect()

# 6. Final Evaluation on strictly unseen hold-out data
print("\n--- Final Evaluation on Test Set ---")
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc:.4f}')

# 7. Extract predictions for Confusion Matrix
test_images, test_labels = zip(*(list(test_dataset.as_numpy_iterator())))
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int).reshape(-1)

# Confusion matrix & Report
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels, target_names=['Benign', 'Malignant']))

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('training_validation_plots.png')