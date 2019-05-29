import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from img_pre import *

batch_size = 64
LAMBDA = 10


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = wgan_af(sample)
        plt.imshow(sample, cmap='Greys_r')

    return fig


real_data = tf.placeholder(dtype=tf.float32, shape=[64, 64, 64, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[64, 100], name='noise')


def lrelu(x, leak=0.2):
    # leakyReLU
    return tf.maximum(x, leak * x)


def discriminator(image, reuse=False):
    stddev = 0.02
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=stddev))
        h0 = lrelu(tf.layers.batch_normalization(h0, training=True))

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=stddev))
        h1 = lrelu(tf.layers.batch_normalization(h1, training=True))

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=stddev))
        h2 = lrelu(tf.layers.batch_normalization(h2, training=True))

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=stddev))
        h3 = tf.nn.sigmoid(tf.layers.batch_normalization(h3, training=True))

        h4 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)  # 全连接层
        return h4


def generator(z, ):
    with tf.variable_scope('generator', reuse=False):
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])

        h0 = tf.nn.relu(tf.layers.batch_normalization(h0, training=True))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=True))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=True))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=True))

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh,
                                        name='g')
        return h4


fake_data = generator(noise)
D_real = discriminator(real_data)
D_fake = discriminator(fake_data, reuse=True)

# 损失函数
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=vars_d)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=vars_g)

max_epoch = 200
input_width = 96
input_height = 96
output_width = 64
output_height = 64
z_dim = 100
is_train = True
data_path = r'./data/faces/*.jpg'

G = []
D = []

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    i = 0
    for epoch in range(max_epoch):
        data = glob(data_path)
        np.random.shuffle(data)
        for idx in range(1, 700):
            batch_files = data[idx * batch_size:(idx + 1) * batch_size]
            batch = [wgan_pre(batch_file, output_width=64, output_height=64) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={real_data: batch_images, noise: batch_z})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={noise: batch_z})
            D = D + [D_loss_curr]
            G = G + [G_loss_curr]

            if idx % 100 == 0:
                print(idx)
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))

            if idx % 100 == 0:
                samples = sess.run(fake_data, feed_dict={noise: batch_z})
                fig = plot(samples)
                plt.savefig('dcgan_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

    x = np.linspace(0, len(D), len(D))
    fig = plt.figure()
    plt.plot(x, D, c='b')
    plt.xlabel('epoch')
    plt.ylabel('D_loss')
    plt.savefig("D_dcgan.jpg")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x, G, c='r')
    plt.xlabel('epoch')
    plt.ylabel('D_loss')
    plt.savefig("G_dcgan.jpg")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x, G, c='b')
    plt.plot(x, D, c='r')
    plt.xlabel('epoch')
    plt.ylabel('D_loss/G_loss')
    plt.savefig("GD_dcgan.jpg")
    plt.close(fig)

    D = np.array(D)
    G = np.array(G)
    np.save('D_dcgan.npy', D)
    np.save('G_dcgan.npy', G)