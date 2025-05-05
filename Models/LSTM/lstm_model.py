# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:52:39 2025

@author: onur_
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import tensorflow as tf

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

warnings.filterwarnings("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except Exception as e:
        print("Could not set memory growth:", e)


newsgroup=fetch_20newsgroups(subset="all")
X=newsgroup.data
y=newsgroup.target

tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
X_sequences=tokenizer.texts_to_sequences(X)
X_padded=pad_sequences(X_sequences,maxlen=100)

X_train,X_test,y_train,y_test=train_test_split(X_padded,y_encoded,test_size=0.2,random_state=42)

def f1_score(y_true,y_pred):
    y_pred=K.round(y_pred)
    
    tp=K.sum(K.cast(y_true*y_pred,"float"),axis=0)
    fp=K.sum(K.cast((1-y_true)*y_pred,"float"),axis=0)
    fn=K.sum(K.cast(y_true*(1-y_pred),"float"),axis=0)
    
    precision=tp/(tp+fp+K.epsilon())
    recall=tp/(tp+fn+K.epsilon())
    
    f1=2*(precision*recall)/(precision+recall+K.epsilon())
    
    return K.mean(f1)

def build_lstm_model():
    
    model=Sequential()

    model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
    model.add(LSTM(units=64,return_sequence=False))
    model.add(Dropout(0.5))
    model.add(Dense(20,activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy",f1_score])
    
    return model

model=build_lstm_model()
model.summary()


early_stopping=EarlyStopping(monitor="val_accuracy",patience=5, restore_best_weights=True)
history=model.fit(X_train,y_train,
                  epochs=5,
                  batch_size=32,
                  validation_split=0.1
                  callbacks=[early_stopinng])

loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test loss: {loss: .4f}, Test accuracy: {accuracy: .4f}")

plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Training accuracy")
plt.plot(history.history["val_accuracy"],label="Validation accuracy")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid("True")










