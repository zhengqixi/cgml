from poopoonet import  poopoonet
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("mnist_data/",  one_hot=True)
train = mnist[0]
validate = mnist[1]
test = mnist[2]
dev = tf.constant(train.images, dtype=tf.float32, shape=[55000, 28, 28, 1])
print(dev)
net = poopoonet.build_model(dev, 10, True)
print(net)