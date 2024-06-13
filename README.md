

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
Epoch 1/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.5375 - loss: 10.5443 - val_accuracy: 0.5000 - val_loss: 9.4307 - learning_rate: 1.0000e-04
Epoch 2/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.6800 - loss: 8.8310 - val_accuracy: 0.6875 - val_loss: 7.8401 - learning_rate: 1.0000e-04
Epoch 3/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7802 - loss: 7.4612 - val_accuracy: 0.8750 - val_loss: 6.5385 - learning_rate: 1.0000e-04
Epoch 4/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7210 - loss: 6.4106 - val_accuracy: 0.7812 - val_loss: 5.7096 - learning_rate: 1.0000e-04
Epoch 5/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7370 - loss: 5.5154 - val_accuracy: 0.7500 - val_loss: 4.9157 - learning_rate: 1.0000e-04
Epoch 6/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8012 - loss: 4.7026 - val_accuracy: 0.7188 - val_loss: 4.3215 - learning_rate: 1.0000e-04
Epoch 7/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7258 - loss: 4.1752 - val_accuracy: 0.7500 - val_loss: 3.6888 - learning_rate: 1.0000e-04
Epoch 8/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7639 - loss: 3.6414 - val_accuracy: 0.7188 - val_loss: 3.4001 - learning_rate: 1.0000e-04
Epoch 9/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7938 - loss: 3.1904 - val_accuracy: 0.5312 - val_loss: 3.1548 - learning_rate: 1.0000e-04
Epoch 10/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7668 - loss: 2.8463 - val_accuracy: 0.8125 - val_loss: 2.6648 - learning_rate: 1.0000e-04
Epoch 11/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7529 - loss: 2.5974 - val_accuracy: 0.7188 - val_loss: 2.5208 - learning_rate: 1.0000e-04
Epoch 12/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7865 - loss: 2.3558 - val_accuracy: 0.8125 - val_loss: 2.1453 - learning_rate: 1.0000e-04
Epoch 13/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7704 - loss: 2.2037 - val_accuracy: 0.7500 - val_loss: 2.1869 - learning_rate: 1.0000e-04
Epoch 14/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7971 - loss: 1.9779 - val_accuracy: 0.7812 - val_loss: 1.8870 - learning_rate: 1.0000e-04
Epoch 15/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7615 - loss: 1.8693 - val_accuracy: 0.6875 - val_loss: 1.8474 - learning_rate: 1.0000e-04
Epoch 16/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7536 - loss: 1.7744 - val_accuracy: 0.7500 - val_loss: 1.7627 - learning_rate: 1.0000e-04
Epoch 17/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7627 - loss: 1.6590 - val_accuracy: 0.7500 - val_loss: 1.6667 - learning_rate: 1.0000e-04
Epoch 18/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7930 - loss: 1.5763 - val_accuracy: 0.8750 - val_loss: 1.4653 - learning_rate: 1.0000e-04
Epoch 19/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7759 - loss: 1.5123 - val_accuracy: 0.7812 - val_loss: 1.4847 - learning_rate: 1.0000e-04
Epoch 20/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8240 - loss: 1.4153 - val_accuracy: 0.8438 - val_loss: 1.3117 - learning_rate: 1.0000e-04
Epoch 21/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8520 - loss: 1.3051 - val_accuracy: 0.8125 - val_loss: 1.3615 - learning_rate: 1.0000e-04
Epoch 22/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.7928 - loss: 1.2871 - val_accuracy: 0.7500 - val_loss: 1.3350 - learning_rate: 1.0000e-04
Epoch 23/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8535 - loss: 1.2180 - val_accuracy: 0.7812 - val_loss: 1.2494 - learning_rate: 1.0000e-04
Epoch 24/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8163 - loss: 1.2232 - val_accuracy: 0.8750 - val_loss: 1.1739 - learning_rate: 1.0000e-04
Epoch 25/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8470 - loss: 1.1600 - val_accuracy: 0.7812 - val_loss: 1.1796 - learning_rate: 1.0000e-04
Epoch 26/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8637 - loss: 1.1121 - val_accuracy: 0.6875 - val_loss: 1.4422 - learning_rate: 1.0000e-04
Epoch 27/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8352 - loss: 1.1499 - val_accuracy: 0.7500 - val_loss: 1.1797 - learning_rate: 1.0000e-04
Epoch 28/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7669 - loss: 1.1451 - val_accuracy: 0.7188 - val_loss: 1.1903 - learning_rate: 5.0000e-05
Epoch 29/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8717 - loss: 1.0368 - val_accuracy: 0.8750 - val_loss: 0.9741 - learning_rate: 5.0000e-05
Epoch 30/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8243 - loss: 1.0574 - val_accuracy: 0.7188 - val_loss: 1.1317 - learning_rate: 5.0000e-05
Epoch 31/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8182 - loss: 1.1024 - val_accuracy: 0.6875 - val_loss: 1.1550 - learning_rate: 5.0000e-05
Epoch 32/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8370 - loss: 1.0183 - val_accuracy: 0.8438 - val_loss: 0.9755 - learning_rate: 5.0000e-05
Epoch 33/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8095 - loss: 1.0475 - val_accuracy: 0.7500 - val_loss: 1.1700 - learning_rate: 2.5000e-05
Epoch 34/50
 9/11 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.8095 - loss: 1.02892024-06-13 21:33:35.256071: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
	 [[{{node IteratorGetNext}}]]
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\lib\contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self.gen.throw(typ, value, traceback)
11/11 ━━━━━━━━━━━━━━━━━━━━ 17s 1s/step - accuracy: 0.8041 - loss: 1.0380 - val_accuracy: 0.8438 - val_loss: 0.9395 - learning_rate: 2.5000e-05
Epoch 35/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 22s 2s/step - accuracy: 0.8643 - loss: 0.9633 - val_accuracy: 0.8125 - val_loss: 1.0024 - learning_rate: 2.5000e-05
Epoch 36/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8209 - loss: 1.0447 - val_accuracy: 0.7188 - val_loss: 1.1423 - learning_rate: 2.5000e-05
Epoch 37/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7989 - loss: 1.0475 - val_accuracy: 0.7812 - val_loss: 1.1297 - learning_rate: 2.5000e-05
Epoch 38/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7944 - loss: 1.0612 - val_accuracy: 0.8438 - val_loss: 0.9342 - learning_rate: 1.2500e-05
Epoch 39/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8287 - loss: 0.9901 - val_accuracy: 0.8125 - val_loss: 1.1044 - learning_rate: 1.2500e-05
Epoch 40/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8006 - loss: 1.0653 - val_accuracy: 0.7188 - val_loss: 1.1950 - learning_rate: 1.2500e-05
Epoch 41/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8599 - loss: 0.9771 - val_accuracy: 0.7812 - val_loss: 1.0377 - learning_rate: 1.2500e-05
Epoch 42/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.7771 - loss: 1.0354 - val_accuracy: 0.8125 - val_loss: 0.9869 - learning_rate: 6.2500e-06
Epoch 43/50
11/11 ━━━━━━━━━━━━━━━━━━━━ 21s 2s/step - accuracy: 0.8027 - loss: 0.9997 - val_accuracy: 0.6562 - val_loss: 1.1453 - learning_rate: 6.2500e-06
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - accuracy: 0.8125 - loss: 0.9755
```

![lossXaccuracyMelanoma](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/270269e8-c894-4680-8cbf-b9adda1aaec2)
