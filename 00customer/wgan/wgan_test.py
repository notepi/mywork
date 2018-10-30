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
from keras.utils import plot_model
from keras.callbacks import TensorBoard

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
    '''
    BatchNormalization:
    参考链接：https://www.cnblogs.com/guoyaohua/p/8724433.html
    ①不仅仅极大提升了训练速度，收敛过程大大加快；
    ②还能增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式
    ，所以不用Dropout也能达到相当的效果；
    ③另外调参过程也简单多了，对于初始化要求没那么高，而且可以使用大的学习率等。
    参数：
    在每一个批次的数据中标准化前一层的激活项， 即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。
    momentum: 移动均值和移动方差的动量。（？？？）
    '''
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
    
    #128*7*7 ？？？
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    #将输入重新调整为特定的尺寸。
    model.add(Reshape((7, 7, 128)))
    '''
    UpSampling2D可以看作是Pooling的反向操作，就是采用Nearest Neighbor interpolation来进行放大，
    说白了就是复制行和列的数据来扩充feature map的大小。反向梯度传播的时候，应该就是每个单元格的梯度的和（猜测）。
    Conv2DTranspose就是正常卷积的反向操作，无需多讲。
    将数据的行和列分别重复size[0]和size[1]次
    参数：size：整数tuple，分别为行和列上采样因子
    '''
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
def train( epochs, batch_size=128, sample_interval=50):

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))

    for epoch in range(epochs):

        for _ in range(n_critic):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            
            # Select a random batch of images
            #产生随机数
            #第一个参数随机数中最小值
            #第二个参数随机数中最大值
            #第三个参数数据的个数
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Generate a batch of new images
            gen_imgs = generator.predict(noise)

            # Train the critic
            d_loss_real = critic.train_on_batch(imgs, valid)
            d_loss_fake = critic.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # Clip critic weights
            # 修剪(Clip)评价层参数
            for l in critic.layers:
                weights = l.get_weights()
                '''
                https://blog.csdn.net/qq1483661204/article/details/78150203
                clip这个函数将将数组中的元素限制在a_min, a_max之间，
                大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
                '''
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)


        # ---------------------
        #  Train Generator
        # ---------------------

        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
#
#        # If at save interval => save generated image samples
#        if epoch % sample_interval == 0:
#            self.sample_images(epoch)

def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 1

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()
    

#保存模型
def save(model, model_name):
    model_path = "saved_model/%s.json" % model_name
    weights_path = "saved_model/%s_weights.hdf5" % model_name
    options = {"file_arch": model_path,
                "file_weight": weights_path}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])
    
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
    
    # The generator takes noise as input and generated imgs
    z = Input(shape=(latent_dim,))
    img = generator(z)
    
    # For the combined model we will only train the generator
    critic.trainable = False
    
    # The critic takes generated images as input and determines validity
    valid = critic(img)
    
    # The combined model  (stacked generator and critic)
    combined = Model(z, valid)
    combined.compile(loss=wasserstein_loss,
        optimizer=optimizer,
        metrics=['accuracy'])
    
#    #train
#    train(epochs=4, batch_size=32, sample_interval=50)
#    plot_model(generator,show_shapes=True, to_file='model.png')
    pass