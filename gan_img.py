import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.misc
import cv2


from glob import glob


# get_image读入图像后直接使用transform函数
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, grayscale = False):
  # if (grayscale):
  #   return scipy.misc.imread(path, flatten = True).astype(np.float)
  # else:
    # Reference: https://github.com/carpedm20/DCGAN-tensorflow/issues/162#issuecomment-315519747
    img_bgr = cv2.imread(path)
    # Reference: https://stackoverflow.com/a/15074748/
    img_rgb = img_bgr[..., ::-1]
    return img_rgb.astype(np.float)


# transform函数
def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
  # 中心crop之后resize
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
  # 直接resize
  #   cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    cropped_image = np.resize(image,[resize_height,resize_width])
    # 标准化处理
  return np.array(cropped_image)/127.5 - 1.

# 中心crop，再进行缩放
def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  # return scipy.misc.imresize(
  return np.resize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

# data_dir = './/data'
# dataset_name = 'faces'
# input_fname_pattern = '*.jpg'

# data = glob(data_path)
#
# img = imread(data[0])

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


#判别模型的输入和参数初始化
X = tf.placeholder(tf.float32, shape=[None, 2304])

D_W1 = tf.Variable(xavier_init([2304, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

#生成模型的输入和参数初始化
Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 2304]))
G_b2 = tf.Variable(tf.zeros(shape=[2304]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

#随机噪声采样函数
def sample_Z(m, n):
    #从-1到1的随机矩阵
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    #tf.matmul 矩阵相乘
    #tf.nn.relu 激活函数
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

#喂入数据 真实 和 虚假
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
#D_loss_real 越小越好？
#tf.ones_like
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

#adam算法的优化器
#动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率
#都有个确定范围，使得参数比较平稳。
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
# Z_dim = 100

# mnist = input_data.read_data_sets('./MNIST', one_hot=True)

batch_size=64
max_epoch = 100
input_width = 96
input_height = 96
output_width = 48
output_height = 48
c_dim = 3
grayscale = False
z_dim = 100
# batch_size=64

# train_size=??

# data_path = os.path.join(data_dir, dataset_name, input_fname_pattern)
data_path = r'./data/faces/*.jpg'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(max_epoch):
    data = glob(data_path)
    np.random.shuffle(data)
    # bath_idxs = min(len(data), train_size)  ??
    bath_idxs = len(data);

    # for idx in range(0,bath_idxs):
    for idx in range(0,64,32000):
        batch_files = data[idx*batch_size:(idx+1)*batch_size]
        batch = [
            get_image(batch_file,
                      input_height=input_height,
                      input_width=input_width,
                      resize_height=output_height,
                      resize_width=output_width,
                      crop=False,
                      grayscale=grayscale) for batch_file in batch_files
        ]
        batch1 = batch
        # batch2 = batch1
        # for i in range(len(batch1)):
        #     batch2[i] = np.resize(batch1,[input_height*input_width,1])

        batch = [np.resize(batch_file,[output_width*output_height]) for batch_file in batch1]
        if grayscale:
            batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
        else:
            batch_images = np.array(batch).astype(np.float32)
        batch_z = np.random.uniform(-1,1,[batch_size,z_dim]).astype(np.float32)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_images, Z: batch_z})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: batch_z})

        # if it % 1000 == 0:
        #     print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
            # print()

# writer = tf.summary.FileWriter('./graph1',tf.get_default_graph())
# writer.close()
# if not os.path.exists('out/'):
#     os.makedirs('out/')

# i = 0
#
# for it in range(1000000):
#     if it % 1000 == 0:
#         samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
#
#         fig = plot(samples)
#         plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#         i += 1
#         plt.close(fig)
#
# #
#     X_mb, _ = mnist.train.next_batch(mb_size)
#

