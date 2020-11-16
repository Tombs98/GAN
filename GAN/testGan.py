from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import pathlib
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


#制作本地数据集
data_path = pathlib.Path("./tupian") #路径
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
random.shuffle(all_image_paths)  # 打散
image_count = len(all_image_paths)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths))
##print(ds)
def load_and_preprocess_from_path_label(path):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    image /= 255.0  # 归一化到[0,1]范围

    return image
image_label_ds  = ds.map(load_and_preprocess_from_path_label)
ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
BATCH_SIZE = 32
ds = ds.batch(BATCH_SIZE)
def change_range(image):
    return 2*image-1

keras_ds = ds.map(change_range)
#print(keras_ds)


class GAN():
    def __init__(self):
        # --------------------------------- #
        #   行192，列192，也就是mnist的shape
        # --------------------------------- #
        self.img_rows = 192
        self.img_cols = 192
        self.channels = 3
        # 28,28,1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        # adam优化器
        optimizer = Adam(0.0002, 0.5)
        # 判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # 生成器
        self.generator = self.build_generator()

        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        # 在训练generate的时候不训练discriminator
        self.discriminator.trainable = False
        # 对生成的假图片进行预测
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        # --------------------------------- #
        #   生成器，输入一串随机数字
        # --------------------------------- #
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # np.prod计算所有元素的乘积，保证生成输出的维度与图片维度一致
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        print("noise", noise)
        print("img", img)
        print("Model", Model(noise,img))
        return Model(noise, img)

    def build_discriminator(self):
        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # 输入一张图片
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, sample_interval=50):

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # 获得数据
        X_train = next(iter(keras_ds))
        X_train = np.expand_dims(X_train, axis=3)
        for epoch in range(epochs):

            # --------------------------- #
            #   随机选取batch_size个图片
            #   对discriminator进行训练
            # --------------------------- #

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            # train_on_batch返回loss(若模型编译时赋予metrics=['acc']则还返回acc)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------------- #
            #  训练generator
            # --------------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("gan_mnist/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./gan_mnist"):
        os.makedirs("./gan_mnist")
    gan = GAN()
    gan.train(epochs=100, batch_size=1, sample_interval=1)

