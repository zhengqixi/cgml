import tensorflow.contrib.slim as slim
import tensorflow as tf
class poopoonet:

    @staticmethod
    def build_model(input, num_output, training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.5)):
            net = slim.layers.convolution(input, num_outputs=20, kernel_size=3, scope='conv1')
            net = slim.layers.max_pool2d(net, 3, scope='pool1')
            net = slim.layers.convolution(net, num_outputs=15, kernel_size= 2, stride=1,scope='conv2')
            net = slim.layers.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.layers.flatten(net, scope='flat')
            net = slim.layers.fully_connected(net, 100)
            net = slim.stack(net, slim.fully_connected, [50, num_output], scope='fc')
            return net

    @staticmethod
    def train_model(train_data, validate_data, outdir, num_output):
        predictions =  poopoonet.build_model(train_data['images'], num_output, False)
        slim.losses.softmax_cross_entropy(predictions, train_data['labels'])
        total_loss = slim.losses.get_total_loss()
        optimizer = tf.train.GradientDescentOptimizer()
        trainer = slim.learning.create_train_op(total_loss, optimizer)
        print(trainer)
