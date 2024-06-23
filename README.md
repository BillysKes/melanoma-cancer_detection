

# Table of Contents

1. [Introduction](#intro)
   - [Project Overview](#overview)
   - [Why Early Detection of Melanoma is important](#)
2. [Dataset Description](#data-descr)
3. [Data Preprocessing](#preprocessing)
   - [Data Augmentation](#)
4. [Implementation](#implementation)
   - [Transfer learning concept](#)
   - [VGG16(Convolution Neural Network)](#)
5. [Evaluation](#)
   - [Confusion Matrix](#)



## 1. Introduction


### Project Overview
Dermatology is the branch of medicine that deals with the skin and one aspect that dermatology deals with is the management of skin diseases such as skin cancer. Computer-aided diagnostic tools can be used for more efficient and accurate detection of skin diseases by combining artificial intelligence and computer vision. This project focus on a dataset consisting of 13.900 images, each uniformly sized at 224 x 224 pixels, specifically designed to enhance the early detection of melanoma. Through the use of pre-trained convolutional neural networks, we aim to develop powerful models capable of distinguishing between benign and malignant skin lesions.

### Why Early Detection of Melanoma is important
Melanoma is the most dangerous type of skin cancer and is known for its high mortality rates. Some of the reasons why early melanoma detection is important are listed below : 

- Improved Prognosis/metastasis prevention : the 5-year survival rate for early-detected melanoma is about 99%, compared to 15-20% for advanced-stage melanoma
- Treatment effectiveness : early-stage melanomas can often be treated effectively with surgical excision alone, whereas advanced melanomas may require chemotherapy, radiation therapy, and immunotherapy
- Enhanced Quality of Life : Patients diagnosed early can avoid the severe symptoms and complications associated with advanced melanoma
- Reduced Healthcare Costs : Early diagnosis and treatment of melanoma can substantially reduce healthcare costs associated with advanced cancer treatments

The improved accuracy of melanoma detection can help doctors identify potential melanomas they might miss during a visual examination, and the increased efficiency can save doctors time and allowing them to focus on patients with concerning lesions flagged by the AI. So for both reasons, AI can lead to an early diagnosis.


## 2. Dataset Description
Information about the dataset used in this project can be found at :  https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data

## 3. Data Preprocessing

### Data Augmentation
Data augmentation is a technique that can be used for image classification to expand the size of a dataset by generating new images from existing ones by flipping, cropping, rotating, shifting, zooming and more.
```

train_datagen = tf.keras.Sequential([
    Rescaling(1./255),  # Rescales pixel values to [0, 1]
    RandomFlip(),  # Random horizontal flip
    RandomRotation(factor=0.25),  # Rotates images randomly up to 20 degrees
#    RandomBrightness(factor=0.25)  # Add random brightness adjustment
    RandomTranslation(height_factor=0.2, width_factor=0.2),  # Shifts images vertically and horizontally up to 20%
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

train_dataset = train_dataset.map(lambda x, y: (train_datagen(x), y)).repeat()
test_dataset = test_dataset.map(lambda x, y: (test_datagen(x), y)).repeat()

```

Data augmentation techniques are specified in the train_datagen and test_datagen in order to generate new images from the existing ones. Rescaling transforms the pixel values to [0,1], RandomFlip horizontally flips an image horizontally or vertically, RandomRotation rotates an image randomly up to 20 degrees, and RandomTranslation shifts an image vertically and horizontally up to 20%. Then, map function is used to create augmented versions of the training images on-the-fly. Lambda function takes each batch of images (x) and their corresponding labels (y), applies the transformations to the images, and returns the transformed images along with their labels. This whole process is performed at the time each image is fed into the model for training and it is called On-the-fly data augmentation.


## 4. Implementation

### Transfer learning

Transfer learning is a technique where in which knowledge of an already trained ML model is re-used to a different but related problem and this knowledge reusability can improve the generalization for that related problem. That technique is commonly used when there aren't enough training data available or when a neural network exists that is already trained(pre-trained) on a similar task, and those networks are usually trained on a huge dataset. Some of the benefits gained from using transfer learning are, better perfomances and the training time reduction. 


### VGG16(Convolutional Neural Network)

VGG16 is a convolutional neural network with 13 convolution layers and 3 fully connected layers, where there are 2 consecutive conv layers(3x3 filter size, 64 filters each, stride 1, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied twice, but, in the second time there are 128 filters. Then, there are 3 consecutive conv layers(3x3 filter size, 256 filters each, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied three times, but, in the second time and third time, there are 512 filters each. Finally, there are 3 consecutive fully connected layers, where the first two layers have 4096 neurons each, and the final layer has 1000 neurons, corresponding to the 1000 classes in the ILSVRC challenge. This CNN can be used as a pretrained model on the ImageNet dataset, for example, and it can be customized to perform a different but simmilar task, which in this case is the classification of medical images.


```
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-3)),
    BatchNormalization(),  # Add Batch Normalization after the first Dense layer
#    Dropout(0.2),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))
])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=5e-6)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

num_train_samples = tf.data.experimental.cardinality(train_dataset).numpy() * BATCH_SIZE
num_test_samples = tf.data.experimental.cardinality(test_dataset).numpy() * BATCH_SIZE
train_steps_per_epoch = num_train_samples // BATCH_SIZE
val_steps_per_epoch = num_test_samples // BATCH_SIZE

history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=test_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks=[early_stopping, reduce_lr]
)



test_loss, test_acc = model.evaluate(test_dataset, steps=val_steps_per_epoch)
print('Test accuracy:', test_acc)
```

The pre-trained VGG16 model is loaded as a base model and it is built with a flatten layer with batch normalization, a fully connected layer of 512 neurons and a final layer(with L2 regularization to this layer and the previous) of a single neuron for binary classification. The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy metrics. Also, early stopping(patience=5) and learning rate reduction callbacks are applied, and the fixed size of epochs are 50.


## 5. Evaluation

```
# Making predictions on the test dataset
test_dataset = test_dataset.take(val_steps_per_epoch) 
test_images, test_labels = zip(*(list(test_dataset.as_numpy_iterator())))
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int).reshape(-1)

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
```

```
Found 11879 files belonging to 2 classes.
Found 2000 files belonging to 2 classes.

Epoch 1/50
743/743 [==============================] - 123s 162ms/step - loss: 0.8639 - accuracy: 0.8103 - val_loss: 0.6890 - val_accuracy: 0.8355 - lr: 1.0000e-04
Epoch 2/50
743/743 [==============================] - 129s 174ms/step - loss: 0.6363 - accuracy: 0.8342 - val_loss: 0.5813 - val_accuracy: 0.8310 - lr: 1.0000e-04
Epoch 3/50
743/743 [==============================] - 120s 161ms/step - loss: 0.5434 - accuracy: 0.8393 - val_loss: 0.5134 - val_accuracy: 0.8425 - lr: 1.0000e-04
Epoch 4/50
743/743 [==============================] - 129s 174ms/step - loss: 0.4995 - accuracy: 0.8373 - val_loss: 0.4624 - val_accuracy: 0.8565 - lr: 1.0000e-04
Epoch 5/50
743/743 [==============================] - 119s 160ms/step - loss: 0.4662 - accuracy: 0.8353 - val_loss: 0.5185 - val_accuracy: 0.7815 - lr: 1.0000e-04
Epoch 6/50
743/743 [==============================] - 126s 170ms/step - loss: 0.4353 - accuracy: 0.8419 - val_loss: 0.5539 - val_accuracy: 0.7555 - lr: 1.0000e-04
Epoch 7/50
743/743 [==============================] - 114s 153ms/step - loss: 0.4226 - accuracy: 0.8406 - val_loss: 0.4542 - val_accuracy: 0.8030 - lr: 1.0000e-04
Epoch 8/50
743/743 [==============================] - 116s 156ms/step - loss: 0.4086 - accuracy: 0.8440 - val_loss: 0.8476 - val_accuracy: 0.6405 - lr: 1.0000e-04
Epoch 9/50
743/743 [==============================] - 126s 169ms/step - loss: 0.3992 - accuracy: 0.8390 - val_loss: 0.5368 - val_accuracy: 0.7595 - lr: 1.0000e-04
Epoch 10/50
743/743 [==============================] - 117s 157ms/step - loss: 0.3990 - accuracy: 0.8382 - val_loss: 0.5561 - val_accuracy: 0.7420 - lr: 1.0000e-04
Epoch 11/50
743/743 [==============================] - 105s 141ms/step - loss: 0.3747 - accuracy: 0.8491 - val_loss: 0.3918 - val_accuracy: 0.8350 - lr: 1.0000e-05
Epoch 12/50
743/743 [==============================] - 125s 169ms/step - loss: 0.3723 - accuracy: 0.8513 - val_loss: 0.3884 - val_accuracy: 0.8320 - lr: 1.0000e-05
Epoch 13/50
743/743 [==============================] - 114s 153ms/step - loss: 0.3685 - accuracy: 0.8523 - val_loss: 0.3639 - val_accuracy: 0.8530 - lr: 1.0000e-05
Epoch 14/50
743/743 [==============================] - 113s 152ms/step - loss: 0.3672 - accuracy: 0.8513 - val_loss: 0.3676 - val_accuracy: 0.8495 - lr: 1.0000e-05
Epoch 15/50
743/743 [==============================] - 105s 141ms/step - loss: 0.3596 - accuracy: 0.8541 - val_loss: 0.3826 - val_accuracy: 0.8365 - lr: 1.0000e-05
Epoch 16/50
743/743 [==============================] - 125s 168ms/step - loss: 0.3594 - accuracy: 0.8507 - val_loss: 0.3752 - val_accuracy: 0.8375 - lr: 1.0000e-05
Epoch 17/50
743/743 [==============================] - 105s 142ms/step - loss: 0.3647 - accuracy: 0.8497 - val_loss: 0.3593 - val_accuracy: 0.8500 - lr: 5.0000e-06
Epoch 18/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3592 - accuracy: 0.8540 - val_loss: 0.3573 - val_accuracy: 0.8515 - lr: 5.0000e-06
Epoch 19/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3584 - accuracy: 0.8545 - val_loss: 0.3841 - val_accuracy: 0.8320 - lr: 5.0000e-06
Epoch 20/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3577 - accuracy: 0.8523 - val_loss: 0.3550 - val_accuracy: 0.8495 - lr: 5.0000e-06
Epoch 21/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3515 - accuracy: 0.8545 - val_loss: 0.3467 - val_accuracy: 0.8560 - lr: 5.0000e-06
Epoch 22/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3544 - accuracy: 0.8545 - val_loss: 0.3442 - val_accuracy: 0.8625 - lr: 5.0000e-06
Epoch 23/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3574 - accuracy: 0.8522 - val_loss: 0.3484 - val_accuracy: 0.8555 - lr: 5.0000e-06
Epoch 24/50
743/743 [==============================] - 125s 169ms/step - loss: 0.3514 - accuracy: 0.8569 - val_loss: 0.3407 - val_accuracy: 0.8620 - lr: 5.0000e-06
Epoch 25/50
743/743 [==============================] - 105s 141ms/step - loss: 0.3533 - accuracy: 0.8518 - val_loss: 0.3513 - val_accuracy: 0.8535 - lr: 5.0000e-06
Epoch 26/50
743/743 [==============================] - 125s 168ms/step - loss: 0.3497 - accuracy: 0.8574 - val_loss: 0.3464 - val_accuracy: 0.8580 - lr: 5.0000e-06
Epoch 27/50
743/743 [==============================] - 114s 153ms/step - loss: 0.3505 - accuracy: 0.8569 - val_loss: 0.3409 - val_accuracy: 0.8585 - lr: 5.0000e-06
Epoch 28/50
743/743 [==============================] - 105s 141ms/step - loss: 0.3512 - accuracy: 0.8571 - val_loss: 0.3323 - val_accuracy: 0.8620 - lr: 5.0000e-06
Epoch 29/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3464 - accuracy: 0.8586 - val_loss: 0.3374 - val_accuracy: 0.8630 - lr: 5.0000e-06
Epoch 30/50
743/743 [==============================] - 116s 156ms/step - loss: 0.3508 - accuracy: 0.8560 - val_loss: 0.3426 - val_accuracy: 0.8565 - lr: 5.0000e-06
Epoch 31/50
743/743 [==============================] - 125s 168ms/step - loss: 0.3484 - accuracy: 0.8558 - val_loss: 0.3397 - val_accuracy: 0.8630 - lr: 5.0000e-06
Epoch 32/50
743/743 [==============================] - 105s 141ms/step - loss: 0.3483 - accuracy: 0.8576 - val_loss: 0.3495 - val_accuracy: 0.8480 - lr: 5.0000e-06
Epoch 33/50
743/743 [==============================] - 117s 157ms/step - loss: 0.3445 - accuracy: 0.8582 - val_loss: 0.3448 - val_accuracy: 0.8555 - lr: 5.0000e-06

Test accuracy: 0.862


Classification Report:
              precision    recall  f1-score   support

      Benign       0.83      0.91      0.87      1000
   Malignant       0.90      0.81      0.85      1000

    accuracy                           0.86      2000
   macro avg       0.87      0.86      0.86      2000
weighted avg       0.87      0.86      0.86      2000

```
![training_validation_plots](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/2b86416c-871d-448f-9586-04ee066472bd)


### Confusion Matrix

![confusion_matrix](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/40386a5f-44ce-44b2-9d59-7263a52cfa85)



