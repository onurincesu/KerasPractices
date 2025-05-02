# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:52:39 2025

@author: onur_
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow.keras.datasets import imdb #dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report,roc_curve,auc
import kerastuner as kt
from kerastuner.tuners import RandomSearch

warnings.filterwarnings("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except Exception as e:
        print("Could not set memory growth:", e)


#most used 10000 words
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)

#padding: padding will make same every comments length.
pad_len=100
x_train=pad_sequences(x_train,maxlen=pad_len)
x_test=pad_sequences(x_test,maxlen=pad_len)

#build model
def rnn_model(hp):
    model=Sequential()
    
    #embedding layer
    model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int("embedding_output",min_value=32,max_value=128,step=32),
                        input_length=pad_len))
    
    #RNN layer
    model.add(SimpleRNN(units=hp.Int("rnn_units",min_value=32,max_value=128, step=32)))
    
    #dropout
    model.add(Dropout(rate=hp.Float("dropout_rate",min_value=0.1,max_value=0.5,step=0.1)))
    
    #output layer
    model.add(Dense(1,activation="sigmoid"))
    
    #model compiling
    model.compile(optimizer=hp.Choice("optimizer",["adam","rmsprop"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy","AUC"])
    
    return model

#hyperparameter search
tuner=RandomSearch(
    rnn_model,#model function
    objective="val_loss", #looks val_loss metrics
    max_trials=4, #tries just 2 model. bigger values makes longer training time
    executions_per_trial=1,#tries 1 learning for every model
    directory="rnn_tuner_directory",
    project_name="imdb_rnn"    
    )

#early_stop

early_stop=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

tuner.search(x_train,y_train,
             epochs=5,
             validation_split=0.2,
             callbacks=[early_stop])


#take best model
best_model=tuner.get_best_models(num_models=1)[0]

#test best model
loss,accuracy,auc_prob=best_model.evaluate(x_test,y_test) #it returns loss,accuracy and AUC in respectively
print(f"Test loss:{loss:.3f},test accuracy{accuracy:.3f},test AUC:{auc_prob:.3f}")

#evaluating model performance
y_pred_prob=best_model.predict(x_test)
y_pred=(y_pred_prob>0.5).astype("int32")

print(classification_report(y_test, y_pred))

fpr,tpr,_=roc_curve(y_test,y_pred_prob) #FPR(false positive rate),TPR(true positive rate)
roc_auc=auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color="darkorange",lw=2,label="ROC Curve(area=%0.2f)"%roc_auc)
plt.plot([0, 1],[0, 1],color="blue",lw=2,linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Receiver Operation Characteristic ROC Curve")
plt.legend(loc="lower right")
plt.show()