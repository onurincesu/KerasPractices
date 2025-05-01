# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:52:39 2025

@author: onur_
"""
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 #dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except Exception as e:
        print("Could not set memory growth:", e)


#load data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

class_labels=["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
fig,axes=plt.subplots(1,5,figsize=(15,10))
for i in range(5):
    axes[i].imshow(x_train[i])
    label=class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
plt.show()

#data normalization.
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

#in dataset there is 10 labels.
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


#data augmentation
datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
    )

datagen.fit(x_train)

#model building
model=Sequential()

model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(62,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax"))

model.summary()


model.compile(optimizer=RMSprop(learning_rate=0.0001,decay=1e-6),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#model training

history=model.fit(datagen.flow(x_train,y_train,batch_size=64),
          epochs=50,validation_data=(x_test,y_test))

#predictions and accuracy rates for every class
y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,  axis=1)
y_true=np.argmax(y_test,axis=1)

report=classification_report(y_true, y_pred_class,target_names=class_labels)
print(report)

plt.figure()
#loss graph
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="train loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid()

#accuracy graph
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="train accuracy")
plt.plot(history.history["val_accuracy"],label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.grid()
