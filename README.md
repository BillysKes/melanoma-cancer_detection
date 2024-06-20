

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

Epoch 1/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 736s 2s/step - accuracy: 0.7543 - loss: 5.2240 - val_accuracy: 0.7875 - val_loss: 4.2004 - learning_rate: 1.0000e-05
Epoch 2/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 726s 2s/step - accuracy: 0.8154 - loss: 3.9580 - val_accuracy: 0.8100 - val_loss: 3.3806 - learning_rate: 1.0000e-05
Epoch 3/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 720s 2s/step - accuracy: 0.8206 - loss: 3.2545 - val_accuracy: 0.8675 - val_loss: 2.8382 - learning_rate: 1.0000e-05
Epoch 4/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 719s 2s/step - accuracy: 0.8307 - loss: 2.8050 - val_accuracy: 0.7525 - val_loss: 2.6913 - learning_rate: 1.0000e-05
Epoch 5/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 727s 2s/step - accuracy: 0.8429 - loss: 2.4838 - val_accuracy: 0.8780 - val_loss: 2.2532 - learning_rate: 1.0000e-05
Epoch 6/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 721s 2s/step - accuracy: 0.8434 - loss: 2.2596 - val_accuracy: 0.8715 - val_loss: 2.0795 - learning_rate: 1.0000e-05
Epoch 7/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8413 - loss: 2.0924 - val_accuracy: 0.8785 - val_loss: 1.9181 - learning_rate: 1.0000e-05
Epoch 8/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8406 - loss: 1.9494 - val_accuracy: 0.8695 - val_loss: 1.8023 - learning_rate: 1.0000e-05
Epoch 9/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8458 - loss: 1.8103 - val_accuracy: 0.8750 - val_loss: 1.6778 - learning_rate: 1.0000e-05
Epoch 10/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 713s 2s/step - accuracy: 0.8391 - loss: 1.7123 - val_accuracy: 0.8895 - val_loss: 1.5698 - learning_rate: 1.0000e-05
Epoch 11/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 719s 2s/step - accuracy: 0.8465 - loss: 1.6003 - val_accuracy: 0.8855 - val_loss: 1.4773 - learning_rate: 1.0000e-05
Epoch 12/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8515 - loss: 1.5084 - val_accuracy: 0.8665 - val_loss: 1.4073 - learning_rate: 1.0000e-05
Epoch 13/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 716s 2s/step - accuracy: 0.8417 - loss: 1.4375 - val_accuracy: 0.8740 - val_loss: 1.3228 - learning_rate: 1.0000e-05
Epoch 14/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8472 - loss: 1.3522 - val_accuracy: 0.8810 - val_loss: 1.2514 - learning_rate: 1.0000e-05
Epoch 15/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 717s 2s/step - accuracy: 0.8491 - loss: 1.2770 - val_accuracy: 0.8925 - val_loss: 1.1696 - learning_rate: 1.0000e-05
Epoch 16/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 716s 2s/step - accuracy: 0.8503 - loss: 1.2182 - val_accuracy: 0.8900 - val_loss: 1.1124 - learning_rate: 1.0000e-05
Epoch 17/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.8554 - loss: 1.1553 - val_accuracy: 0.8920 - val_loss: 1.0529 - learning_rate: 1.0000e-05
Epoch 18/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 716s 2s/step - accuracy: 0.8508 - loss: 1.1025 - val_accuracy: 0.8835 - val_loss: 1.0134 - learning_rate: 1.0000e-05
Epoch 19/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 715s 2s/step - accuracy: 0.8509 - loss: 1.0614 - val_accuracy: 0.8895 - val_loss: 0.9634 - learning_rate: 1.0000e-05
Epoch 20/20
372/372 ━━━━━━━━━━━━━━━━━━━━ 3084s 8s/step - accuracy: 0.8554 - loss: 1.0026 - val_accuracy: 0.8955 - val_loss: 0.9149 - learning_rate: 1.0000e-05
63/63 ━━━━━━━━━━━━━━━━━━━━ 103s 2s/step - accuracy: 0.8870 - loss: 0.9231
Test accuracy: 0.89550
```

![vgg16_20epochs](https://github.com/BillysKes/melanoma-cancer_detection/assets/73298709/488cf79d-ae9b-49ca-b55d-d20504794fbf)

### Confusion Matrix


