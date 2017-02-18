import tensorflow.contrib.slim as slim
import tensorflow as tf
class poopoonet:

    @staticmethod
    def build_model(input, num_output, training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.5)):
            net = slim.layers.conv2d(input, 20, 3, scope='conv1')
            net = slim.layers.max_pool2d(net, 3, scope='pool1')
            net = slim.layers.conv2d(net, 15, stride=1,scope='conv2')
            net = slim.layers.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.layers.flatten(net, scope='flat')
            net = slim.stack(net, slim.fully_connected, [150, 100 ], scope='fc')
            net = slim.layers.dropout(net, keep_prob=0.75, is_training=training)
            net = slim.stack(net, slim.fully_connected, [50, num_output], scope='fc')
            return net

    @staticmethod
    def train_model(images_train, labels_train, images_validate, labels_validate, outdir, num_output):
        predictions =  poopoonet.build_model(images_train, num_output, False)
        slim.losses.softmax_cross_entropy(predictions, labels_train)
        total_loss = slim.losses.get_total_loss()
        optimizer = tf.train.GradientDescentOptimizer()
        trainer = slim.learning.create_train_op(total_loss, optimizer )
