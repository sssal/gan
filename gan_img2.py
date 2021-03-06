import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from glob import glob

from img_pre import *


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # sample = sample*255
        sample = af_processing(sample)
        plt.imshow(sample, cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# 判别模型的输入和参数初始化
X = tf.placeholder(tf.float32, shape=[None, 2304])
D_W1 = tf.Variable(xavier_init([2304, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

# 生成模型的输入和参数初始化
Z = tf.placeholder(tf.float32, shape=[None, 100])
G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 2304]))
G_b2 = tf.Variable(tf.zeros(shape=[2304]))
theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    # tf.matmul 矩阵相乘
    # tf.nn.relu 激活函数
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# adam算法的优化器
# 动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率
# 都有个确定范围，使得参数比较平稳。
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

batch_size = 64
max_epoch = 100
input_width = 96
input_height = 96
output_width = 48
output_height = 48
z_dim = 100

data_path = r'./data/faces/*.jpg'

if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    i = 0
    for epoch in range(max_epoch):
        data = glob(data_path)
        np.random.shuffle(data)
        for idx in range(1, 700):
            batch_files = data[idx * batch_size:(idx + 1) * batch_size]
            batch = [pre_processing(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)  # 真实样本
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_images, Z: batch_z})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: batch_z})

            if (idx % 100 == 0):
                print(idx)
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))

            if (idx % 100 == 0):
                samples = sess.run(G_sample, feed_dict={Z: batch_z})
                fig = plot(samples)
                plt.savefig('out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
