import datetime

import tensorflow as tf
from tensorflow._api.v2.v2 import optimizers
from tensorflow.python.keras import datasets, layers, Sequential

from TensorBoardUtils import image_grid, plot_to_image


# ##################### 1. 安装 #####################
# pip install tensorboard
# 启动
# tensorboard --logdir logs


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz, drop_remainder=True)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))
network.summary()

optimizer = optimizers.Adam(lr=0.01)

# ##################### 2. 创建日志文件 #####################
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# ##################### 3. 显示数据 #####################
# ##################### 3.1.1 显示一个图片 #####################
# # get x from (x,y)
sample_img = next(iter(db))[0]
# get first image instance
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image("Training sample:", sample_img, step=0)

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28 * 28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b]
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 100 == 0:
        print(step, 'loss:', float(loss))
        # ##################### 3.2 显示折线数据 #####################
        with summary_writer.as_default():
            tf.summary.scalar('train-loss', float(loss), step=step)

            # evaluate
    if step % 500 == 0:
        total, total_correct = 0., 0

        for _, (x, y) in enumerate(ds_val):
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # [b, 784] => [b, 10]
            out = network(x)
            # [b, 10] => [b] 
            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # bool type 
            correct = tf.equal(pred, y)
            # bool tensor => int tensor => numpy
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x.shape[0]

        print(step, 'Evaluate Acc:', total_correct / total)

        # print(x.shape)
        val_images = x[:25]
        val_images = tf.reshape(val_images, [-1, 28, 28, 1])
        with summary_writer.as_default():
            # 单个图片打印
            tf.summary.scalar('test-acc', float(total_correct / total), step=step)
            tf.summary.image("val-onebyone-images:", val_images, max_outputs=25, step=step)

            # ##################### 3.1.2 多个图片合并成一张图片打印 #####################
            val_images = tf.reshape(val_images, [-1, 28, 28])
            figure = image_grid(val_images)
            tf.summary.image('val-images:', plot_to_image(figure), step=step)
