

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

## 3. Data Preprocessing & augmentation

Data augmentation expands the size of a dataset by generating new images from existing ones on-the-fly, which helps the neural network generalize better to unseen skin lesions. 

To prevent data leakage, the data is strictly partitioned. Augmentations (like random rotations and flips) are applied *only* to the training set. The validation and test sets are strictly isolated and only receive standard rescaling and resizing to ensure the model's accuracy is evaluated on pure, unaltered real-world images.

```

# Augmentation for Training Data
train_datagen = tf.keras.Sequential([
    Resizing(height=224, width=224, interpolation="bilinear"),
    RandomFlip(),  
    RandomRotation(factor=0.25),  
    Rescaling(1./255)
])

# Strict Rescaling for Validation/Test Data
test_datagen = tf.keras.Sequential([
    Resizing(height=224, width=224, interpolation="bilinear"), 
    Rescaling(1./255)
])

# 80/20 Train/Validation Split
train_dataset = image_dataset_from_directory(
    "Melanoma Cancer Image Dataset/train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=16,  
    label_mode="binary"  
)

```


## 4. Implementation

### Transfer learning

Transfer learning is a technique where in which knowledge of an already trained ML model is re-used to a different but related problem and this knowledge reusability can improve the generalization for that related problem. That technique is commonly used when there aren't enough training data available or when a neural network exists that is already trained(pre-trained) on a similar task, and those networks are usually trained on a huge dataset. Some of the benefits gained from using transfer learning are, better perfomances and the training time reduction. 


### VGG16(Convolutional Neural Network)

VGG16 is a convolutional neural network with 13 convolution layers and 3 fully connected layers, where there are 2 consecutive conv layers(3x3 filter size, 64 filters each, stride 1, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied twice, but, in the second time there are 128 filters. Then, there are 3 consecutive conv layers(3x3 filter size, 256 filters each, same padding) followed by 1 max pooling layer(2x2, stride 2), and this building block is applied three times, but, in the second time and third time, there are 512 filters each. Finally, there are 3 consecutive fully connected layers, where the first two layers have 4096 neurons each, and the final layer has 1000 neurons, corresponding to the 1000 classes in the ILSVRC challenge. This CNN can be used as a pretrained model on the ImageNet dataset, for example, and it can be customized to perform a different but simmilar task, which in this case is the classification of medical images.


```
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

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

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

```

The script is executed on wsl2(Windows Subsystem for Linux) environment in order to leverage the GPU and speed up the training of the model. The pre-trained VGG16 model is loaded as a base model and it is built with a flatten layer with batch normalization, a fully connected layer of 512 neurons and a final layer(with L2 regularization to this layer and the previous) of a single neuron for binary classification. The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy metrics. Also, early stopping(patience=5) and learning rate reduction callbacks are applied, and the fixed size of epochs are 50.


## 5. Evaluation & Results

The model was evaluated strictly on 2,000 unseen hold-out test images to simulate real-world diagnostic performance.


```

--- Final Evaluation on Test Set ---
125/125 ━━━━━━━━━━━━━━━━━━━━ 14s 111ms/step - accuracy: 0.8975 - loss: 0.2815
Test accuracy: 0.8975

Classification Report:
              precision    recall  f1-score   support

      Benign       0.87      0.93      0.90      1000
   Malignant       0.93      0.86      0.89      1000

    accuracy                           0.90      2000
   macro avg       0.90      0.90      0.90      2000
weighted avg       0.90      0.90      0.90      2000

```


### Learning curces & Confusion Matrix

<img width="1200" height="400" alt="training_validation_plots" src="https://github.com/user-attachments/assets/6b3e665f-d76d-4716-bcdf-177b8def2cac" />


<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/02c0ed87-14ae-416b-a38e-bcadcfc2275c" />


