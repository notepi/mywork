# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:53:26 2018

@author: 潘鹏
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)
def build_critic():

    model = Sequential()
    '''
    filters=16
    filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）
    kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。
    可以是一个整数，为所有空间维度指定相同的值。
    strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。 
    指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
    padding: "valid" 或 "same" ,该参数是大小写敏感的。
    具体解释见：https://blog.csdn.net/fireflychh/article/details/73743849
    '''
    #
    #kernel_size=3
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
def build_generator():

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    #元组中只包含一个元素时，需要在元素后面添加逗号
    #输入有latent_dim个整数的序列，每个序列大小不确定
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)
if __name__ == '__main__':
    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)
    latent_dim = 100
    
    # Following parameter and optimizer set as recommended in paper
    n_critic = 5
    clip_value = 0.01
    optimizer = RMSprop(lr=0.00005)
    
    # Build and compile the critic
    critic = build_critic()
    #损失函数自定义
    #评价标准是：accuracy
    critic.compile(loss=wasserstein_loss,
        optimizer=optimizer,
        metrics=['accuracy'])
    
    # Build the generator
    generator = build_generator()
    
#    # The generator takes noise as input and generated imgs
#    z = Input(shape=(self.latent_dim,))
#    img = self.generator(z)
#    
#    # For the combined model we will only train the generator
#    self.critic.trainable = False
#    
#    # The critic takes generated images as input and determines validity
#    valid = self.critic(img)
#    
#    # The combined model  (stacked generator and critic)
#    self.combined = Model(z, valid)
#    self.combined.compile(loss=self.wasserstein_loss,
#        optimizer=optimizer,
#        metrics=['accuracy'])
    pass