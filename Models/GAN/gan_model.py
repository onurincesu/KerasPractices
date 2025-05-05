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

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam


from tqdm import tqdm

warnings.filterwarnings("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth enabled")
    except Exception as e:
        print("Could not set memory growth:", e)


(x_train,_),(_,_)=mnist.load_data()

x_train=x_train/255
x_train=np.expand_dims(x_train,axis=-1)

z_dim = 100 

def build_discriminator():
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same", input_shape = (28,28,1)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Conv2D(128, kernel_size=3, strides = 2, padding="same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Flatten())
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5), metrics = ["accuracy"])
    
    return model

def build_generator():
    
    model = Sequential()
    
    model.add(Dense(7*7*128, input_dim=z_dim)) 
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Reshape((7,7,128)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=3, strides = 2, padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = "same", activation = "tanh"))
    
    return model
    

def build_gan(generator, discriminator):
    
    discriminator.trainable = False 
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator) 
    model.compile(loss="binary_crossentropy", optimizer = Adam(0.0002, 0.5))
    
    return model
   
discriminator = build_discriminator()
generator = build_generator()
gan = build_gan(generator, discriminator)

print(gan.summary())
epochs = 10000
batch_size = 64 
half_batch = batch_size // 2

# egitim dongusu
for epoch in tqdm(range(epochs), desc = "Training Process"):
    

    idx = np.random.randint(0, x_train.shape[0], half_batch)
    real_images = x_train[idx]
    real_labels = np.ones((half_batch, 1))
    
    noise = np.random.normal(0, 1, (half_batch, z_dim))
    fake_images = generator.predict(noise, verbose = 0)
    fake_labels = np.zeros((half_batch, 1))
    
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
    
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)
        
    if epoch % 100 == 0:
        print(f"\n{epoch}/{epochs} D loss: {d_loss[0]}, G loss: {g_loss}")
        


def plot_generated_images(generator, epoch, examples = 10, dim=(1,10)):
    
    noise = np.random.normal(0, 1, (examples, z_dim)) 
    gen_images = generator.predict(noise, verbose = 0)
    gen_images = 0.5*gen_images + 0.5
    
    plt.figure(figsize = (10,1))
    for i in range(gen_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(gen_images[i, :,:,0], cmap = "gray")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

plot_generated_images(generator, epochs)
