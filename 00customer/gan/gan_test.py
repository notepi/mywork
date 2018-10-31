# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:05:17 2018

@author: pan
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import pandas as pd
import numpy as np

#构造描述器
def build_discriminator():
    #序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”，是一个神经网络的整体，
    #相当于是一个模型的容器，
    #可以往里面添加各种网络层
    model = Sequential()
    
    #Flatten层用来将输入“压平”，即把多维的输入一维化
    #常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    '''
    eg.
    model.add(Convolution2D(64, 3, 3,
            border_mode='same',
            input_shape=(3, 32, 32)))
    # now: model.output_shape == (None, 64, 32, 32)
    
    model.add(Flatten())
    # now: model.output_shape == (None, 65536)
    '''
    model.add(Flatten(input_shape=img_shape))
    #Dense就是常用的全连接层，所实现的运算
    #512指的是神经元的个数
    model.add(Dense(512))
    #对生成的值，使用激活函数重新映射
    '''
    见url：https://www.cnblogs.com/chamie/p/8665251.html
    '''
    '''
    sigmoid和tanh是“饱和激活函数”，而ReLU及其变体则是“非饱和激活函数”。使用“非饱和激活函数”的优势在于两点：
    1.首先，“非饱和激活函数”能解决所谓的“梯度消失”问题。
    2.其次，它能加快收敛速度。
    '''
    '''
    ReLU
    ReLU函数代表的的是“修正线性单元”，它是带有卷积图像的输入x的最大函数(x,o)。
    ReLU函数将矩阵x内所有负值都设为零，其余的值不变。
    ReLU函数的计算是在卷积之后进行的，因此它与tanh函数和sigmoid函数一样，同属于“非线性激活函数”。
    这一内容是由Geoff Hinton首次提出的。
    '''
    '''
    ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。
    Leaky ReLU激活函数是在声学模型（2013）中首次提出的。
    '''
    '''
    url:https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/#leakyrelu
    LeakyRelU是修正线性单元（Rectified Linear Unit，ReLU）的特殊版本，当不激活时，
    LeakyReLU仍然会有非零输出值，从而获得一个小梯度，避免ReLU可能出现的神经元“死亡”现象。
    '''
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    #前面是为了避免细胞失活等用的高级激活函数，所以要专门做一个层
    #到最后了，是单独的一层，激活函数在预定义激活函数中，无需专门做一层
#    #最后一层表示输出的只有一个
#    model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    #打印出模型概况，它实际调用的是keras.utils.print_summary
    model.summary()

    '''
    Keras有两种类型的模型，序贯模型（Sequential）和函数式模型（Model），
    函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况。
    '''
    #构造模型参数，对模型进行封装
    #构造输入的参数
    img = Input(shape=img_shape)
    #构造输入的相关张量的数据
    validity = model(img)
    '''
    利用函数式 API，可以轻易地重用训练好的模型：可以将任何模型看作是一个层，
    然后通过传递一个张量来调用它。
    注意，在调用模型时，您不仅重用模型的结构，还重用了它的权重。
    '''
    #把Sequential()重用，封装在Model内
    return Model(img, validity)
#构造生成器
def build_generator():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    #批量标准化层 (Ioffe and Szegedy, 2014)。
    #在每一个批次的数据中标准化前一层的激活项， 即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。
    model.add(BatchNormalization(momentum=0.8))
    
    for i in range(1):
    
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        pass
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    '''
    np.prod():
        返回给定轴上的数组元素的乘积。
        eg.:
            a=[1,2,3]
            b=np.prod(a):b=1*2*3     
    '''
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    #将输入重新调整为特定的尺寸。
    model.add(Reshape(img_shape))
    #打印出模型概况，它实际调用的是keras.utils.print_summary
    model.summary()
    #元组中只包含一个元素时，需要在元素后面添加逗号
    #输入有latent_dim个整数的序列，每个序列大小不确定？？？
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)
def train(epochs, batch_size=128, sample_interval=50):

#    # Load the dataset
#    (X_train, _), (_, _) = mnist.load_data()
    global X_train


    # Adversarial ground truths
    #生成矩阵：batch_sizeX1维
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        #产生随机数
        #第一个参数随机数中最小值
        #第二个参数随机数中最大值
        #第三个参数数据的个数
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        #从训练数据中，随机获取 X_train.shape[0]个数据
        imgs = X_train[idx]
        
        #随机产生噪声
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        #利用真实数据对描述器进行训练
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        #利用生成数据对描述器进行训练
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        #计算真实数据的损失和生成数据的损失
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        
        #随机产生噪声
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Train the generator (to have the discriminator label samples as valid)
        # G和D是揉在一起了，让G去欺骗D，因为揉在一起了，所以用和在一起的训练器
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
#            sample_images(epoch)
            print("train",epoch)
            pass
            
def sample_images(epoch):
    r, c = 5, 5
    #生成r*c个样本
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    #开始训练的时候对数据做了标准化，准备复原
    # Rescale images 0 - 1
    gen_imgs = (gen_imgs + 1)*127.5
#
#    fig, axs = plt.subplots(r, c)
#    cnt = 0
#    for i in range(r):
#        for j in range(c):
#            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
#            axs[i,j].axis('off')
#            cnt += 1
#    fig.savefig("images/%d.png" % epoch)
#    plt.close()
if __name__ == '__main__':
    AllData = pd.read_csv('../code/dataprocess/finaldata0922.csv')
#    del AllData['time']
    X_train=AllData.iloc[:,:-1].values
#    # Rescale -1 to 1
#    X_train = X_train / 127.5 - 1.
    #增加维度,从第几个括号
    X_train = np.expand_dims(X_train, axis=2)
    
    img_rows = len(AllData.iloc[1,:-1])
    channels = 1
    img_shape = (img_rows, channels)
    latent_dim = 100
     
    optimizer = Adam(0.0002, 0.5)
    
    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    
    # Build the generator
    generator = build_generator()
    
    # The generator takes noise as input and generates imgs
    # 随机产生噪声作为初始值
    z = Input(shape=(latent_dim,))
    img = generator(z)
    
    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(img)

    # The combined model  (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    #将G和D合并在一个model中？？？
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    #train
    train(epochs=3, batch_size=32, sample_interval=200)
    
    pass
