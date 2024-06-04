

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
```
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
