

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
zoom_factor = 0.2
crop_height = int(224 * (1 - zoom_factor))
crop_width = int(224 * (1 - zoom_factor))

train_datagen = tf.keras.Sequential([
    Rescaling(1./255),  # Rescales pixel values to [0, 1]
    RandomFlip("horizontal"),  # Random horizontal flip
    RandomRotation(factor=0.2),  # Rotates images randomly up to 20 degrees
    RandomTranslation(height_factor=0.2, width_factor=0.2)  # Shifts images vertically and horizontally up to 20%

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

train_dataset = train_dataset.map(lambda x, y: (train_datagen(x), y))
test_dataset = test_dataset.map(lambda x, y: (test_datagen(x), y))

```

Data augmentation techniques are specified in the train_datagen and test_datagen in order to generate new images from the existing ones. Rescaling transforms the pixel values to [0,1], RandomFlip horizontally flips an image, RandomRotation rotates an image randomly up to 20 degrees, RandomTranslation shifts an image vertically and horizontally up to 20% and RandomCrop randomly removes sections of the image. Then, map function is used to apply the data augmentation transformations(train_datagen and test_datagen) to the the train-test datasets. Lambda function takes each batch of images (x) and their corresponding labels (y), applies the transformations to the images, and returns the transformed images along with their labels. This whole process is performed at the time each image is fed into the model for training and it is called On-the-fly data augmentation.


## 4. Implementation

### Transfer learning

Transfer learning is a technique where in which knowledge of an already trained ML model is re-used to a different but related problem and this knowledge reusability can improve the generalization for that related problem. That technique is commonly used when there aren't enough training data available or when a neural network exists that is already trained(pre-trained) on a similar task, and those networks are usually trained on a huge dataset. Some of the benefits gained from using transfer learning are, better perfomances and the training time reduction. 


### VGG16(Convolutional Neural Network)

VGG16 is a convolutional neural network with 13 convolution layers and 3 fully connected layers, where there are 2 consecutive conv layers(3x3 filter size, 64 filters each, stride 1, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied twice, but, in the second time there are 128 filters. Then, there are 3 consecutive conv layers(3x3 filter size, 256 filters each, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied three times, but, in the second time and third time, there are 512 filters each. Finally, there are 3 consecutive fully connected layers, where the first two layers have 4096 neurons each, and the final layer has 1000 neurons, corresponding to the 1000 classes in the ILSVRC challenge. This CNN can be used as a pretrained model on the ImageNet dataset, for example, and it can be customized to perform a different but simmilar task, which in this case is the classification of medical images.


```
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

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

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

train_steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy() // BATCH_SIZE
val_steps_per_epoch = tf.data.experimental.cardinality(test_dataset).numpy() // BATCH_SIZE
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=test_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_dataset, steps=val_steps_per_epoch)
print('Test accuracy:', test_acc)
```


## 5. Evaluation

```
Found 11879 files belonging to 2 classes.
Found 2000 files belonging to 2 classes.

Epoch 1/50
743/743 [==============================] - 132s 175ms/step - loss: 0.8795 - accuracy: 0.8087 - val_loss: 0.8754 - val_accuracy: 0.7360 - lr: 1.0000e-04
Epoch 2/50
743/743 [==============================] - 120s 161ms/step - loss: 0.6589 - accuracy: 0.8315 - val_loss: 0.5701 - val_accuracy: 0.8515 - lr: 1.0000e-04
Epoch 3/50
743/743 [==============================] - 129s 174ms/step - loss: 0.5744 - accuracy: 0.8376 - val_loss: 0.4867 - val_accuracy: 0.8720 - lr: 1.0000e-04
Epoch 4/50
743/743 [==============================] - 128s 173ms/step - loss: 0.5211 - accuracy: 0.8321 - val_loss: 0.5868 - val_accuracy: 0.7685 - lr: 1.0000e-04
Epoch 5/50
743/743 [==============================] - 128s 172ms/step - loss: 0.4782 - accuracy: 0.8385 - val_loss: 0.4977 - val_accuracy: 0.8110 - lr: 1.0000e-04
Epoch 6/50
743/743 [==============================] - 127s 170ms/step - loss: 0.4506 - accuracy: 0.8437 - val_loss: 0.5660 - val_accuracy: 0.7595 - lr: 1.0000e-04
Epoch 7/50
743/743 [==============================] - 123s 165ms/step - loss: 0.4305 - accuracy: 0.8461 - val_loss: 0.4391 - val_accuracy: 0.8355 - lr: 1.0000e-05
Epoch 8/50
743/743 [==============================] - 107s 144ms/step - loss: 0.4148 - accuracy: 0.8526 - val_loss: 0.4227 - val_accuracy: 0.8415 - lr: 1.0000e-05
Epoch 9/50
743/743 [==============================] - 125s 168ms/step - loss: 0.4161 - accuracy: 0.8492 - val_loss: 0.4356 - val_accuracy: 0.8240 - lr: 1.0000e-05
Epoch 10/50
743/743 [==============================] - 113s 153ms/step - loss: 0.4055 - accuracy: 0.8507 - val_loss: 0.3868 - val_accuracy: 0.8585 - lr: 1.0000e-05
Epoch 11/50
743/743 [==============================] - 104s 140ms/step - loss: 0.4060 - accuracy: 0.8484 - val_loss: 0.4104 - val_accuracy: 0.8425 - lr: 1.0000e-05
Epoch 12/50
743/743 [==============================] - 125s 168ms/step - loss: 0.4012 - accuracy: 0.8475 - val_loss: 0.3931 - val_accuracy: 0.8500 - lr: 1.0000e-05
Epoch 13/50
743/743 [==============================] - 114s 152ms/step - loss: 0.3975 - accuracy: 0.8491 - val_loss: 0.4036 - val_accuracy: 0.8350 - lr: 1.0000e-05
Epoch 14/50
743/743 [==============================] - 104s 140ms/step - loss: 0.3827 - accuracy: 0.8571 - val_loss: 0.3640 - val_accuracy: 0.8695 - lr: 5.0000e-06
Epoch 15/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3833 - accuracy: 0.8600 - val_loss: 0.3658 - val_accuracy: 0.8590 - lr: 5.0000e-06
Epoch 16/50
743/743 [==============================] - 125s 168ms/step - loss: 0.3889 - accuracy: 0.8520 - val_loss: 0.3707 - val_accuracy: 0.8555 - lr: 5.0000e-06
Epoch 17/50
743/743 [==============================] - 104s 140ms/step - loss: 0.3873 - accuracy: 0.8496 - val_loss: 0.3617 - val_accuracy: 0.8645 - lr: 5.0000e-06
Epoch 18/50
743/743 [==============================] - 124s 168ms/step - loss: 0.3816 - accuracy: 0.8539 - val_loss: 0.3688 - val_accuracy: 0.8535 - lr: 5.0000e-06
Epoch 19/50
743/743 [==============================] - 104s 140ms/step - loss: 0.3761 - accuracy: 0.8542 - val_loss: 0.3591 - val_accuracy: 0.8660 - lr: 5.0000e-06
Epoch 20/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3706 - accuracy: 0.8600 - val_loss: 0.3505 - val_accuracy: 0.8725 - lr: 5.0000e-06
Epoch 21/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3803 - accuracy: 0.8538 - val_loss: 0.3574 - val_accuracy: 0.8640 - lr: 5.0000e-06
Epoch 22/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3839 - accuracy: 0.8497 - val_loss: 0.3545 - val_accuracy: 0.8615 - lr: 5.0000e-06
Epoch 23/50
743/743 [==============================] - 115s 155ms/step - loss: 0.3764 - accuracy: 0.8538 - val_loss: 0.3532 - val_accuracy: 0.8645 - lr: 5.0000e-06
Epoch 24/50
743/743 [==============================] - 124s 168ms/step - loss: 0.3764 - accuracy: 0.8526 - val_loss: 0.3640 - val_accuracy: 0.8550 - lr: 5.0000e-06
Epoch 25/50
743/743 [==============================] - 113s 153ms/step - loss: 0.3680 - accuracy: 0.8585 - val_loss: 0.3557 - val_accuracy: 0.8635 - lr: 5.0000e-06

Test accuracy: 0.87250


```

![training_validation_plots](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/e60d8803-4933-4eee-ba09-7957195ca787)


### Confusion Matrix


![confusion_matrix](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/92589bed-6c6e-478b-bd40-05e5eba9e3af)
