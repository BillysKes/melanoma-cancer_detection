

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
   - [ROC Curve and AUC](#)





## 2. Dataset Description
https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data

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
Epoch 1/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.6135 - loss: 13.7570 - val_accuracy: 0.5938 - val_loss: 7.1305
Epoch 2/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7898 - loss: 4.1197 - val_accuracy: 0.8125 - val_loss: 5.0239
Epoch 3/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8149 - loss: 4.3832 - val_accuracy: 0.8125 - val_loss: 1.8572
Epoch 4/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.7848 - loss: 1.6234 - val_accuracy: 1.0000 - val_loss: 0.0948
Epoch 5/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 26s 2s/step - accuracy: 0.7618 - loss: 1.0242 - val_accuracy: 0.9062 - val_loss: 0.2675
Epoch 6/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.8350 - loss: 0.4563 - val_accuracy: 0.8750 - val_loss: 0.3443
Epoch 7/10
11/11 ━━━━━━━━━━━━━━━━━━━━ 25s 2s/step - accuracy: 0.8185 - loss: 0.3682 - val_accuracy: 0.8125 - val_loss: 0.2810
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - accuracy: 0.9375 - loss: 0.3001
Test accuracy: 0.9375
```

![lossXaccuracyMelanoma](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/270269e8-c894-4680-8cbf-b9adda1aaec2)
