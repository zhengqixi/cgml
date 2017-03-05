import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import datetime
import random


class poopoonet:
    def __init__(self, input_shape, num_output, restore_dir=None):
        self.sess = tf.Session()
        if restore_dir is not None:
            monster_reborn = tf.train.Saver()
            monster_reborn.restore(self.sess, restore_dir)
        else:
            self.build_model(input_shape, num_output)

    def build_model(self, input_shape, num_output):
        # input_shape: [dimension1, dimension2, channels]
        self.inputs = tf.placeholder(tf.float32, shape=[None] + input_shape,
                                     name='input')
        self.num_output = num_output
        self.input_shape = input_shape
        self.keep_prob = tf.placeholder(tf.float32)
        with slim.arg_scope([slim.layers.convolution, slim.layers.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(),
                                                                                     uniform=True),
                            weights_regularizer=slim.l2_regularizer(1e-3)):
            net = slim.layers.convolution(self.inputs, num_outputs=30, kernel_size=3, scope='conv1')
            net = slim.layers.max_pool2d(net, 3, scope='pool1')
            net = slim.layers.convolution(net, num_outputs=40, kernel_size=3, stride=1, scope='conv2')
            net = slim.layers.max_pool2d(net, 2, stride=1, scope='pool2')
            net = slim.layers.flatten(net, scope='flat')
            net = slim.layers.fully_connected(net, 256)
            net = slim.layers.dropout(net, keep_prob=self.keep_prob)
            net = slim.layers.fully_connected(net, 128)
            net = slim.layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None)
            self.net = net

    def train_model(self, learning_rate, keep_prob, train_data, batch_size, train_epoch, validation_data,
                    validation_epoch, outdir=None):
        self.labels = tf.placeholder(tf.float32, shape=[None, self.num_output])
        predict_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.net, labels=self.labels)
        weight_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        total_loss = predict_loss + weight_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.trainer = slim.learning.create_train_op(total_loss, optimizer)
        self.sess.run(tf.global_variables_initializer())
        data_tuple = self.reshape_data(train_data)
        loss = 0
        losses_vector = []
        epoch_vector = []
        validation_accuracy = []
        validation_epoch_time = []
        for epoch in range(1, train_epoch + 1):
            image, label = self.get_batch(data_tuple, batch_size)
            _, loss = self.sess.run([self.trainer, total_loss],
                                    feed_dict={self.inputs: image, self.labels: label, self.keep_prob: keep_prob})
            losses_vector.append(float(np.mean(loss)))
            epoch_vector.append(epoch)
            if epoch % validation_epoch == 0:
                predictions = self.infer_model(validation_data)
                validation_accuracy.append(self.accuracy(predictions, validation_data['labels']))
                validation_epoch_time.append(epoch)
                print("Validation at : ", datetime.datetime.now(), "epoch ", epoch, "with accuracy ",
                      validation_accuracy[len(validation_accuracy) - 1])

        if outdir is not None:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, outdir)
            print("Model saved at: " + save_path)

        return epoch_vector, losses_vector, validation_epoch_time, validation_accuracy

    def reshape_data(self, data):
        data_tuple = []
        for image, label in zip(data['images'], data['labels']):
            data_tuple.append((np.reshape(image, (self.input_shape[0], self.input_shape[1], self.input_shape[2])),
                               np.reshape(label, self.num_output)))
        return data_tuple

    def get_batch(self, data, batch_size):
        batch = random.sample(data, batch_size)
        batch_image = [x[0] for x in batch]
        batch_label = [x[1] for x in batch]
        return batch_image, batch_label

    def accuracy(self, predictions, labels):
        num_correct = 0
        num_total = len(labels)
        for prediction, label in zip(predictions, labels):
            if np.argmax(prediction) == np.argmax(label):
                num_correct += 1
        return num_correct / num_total

    def infer_model(self, data):
        data_tuple = self.reshape_data(data)
        batch_images = [x[0] for x in data_tuple]
        predictions = self.sess.run(tf.nn.softmax(logits=self.net), feed_dict={self.inputs: batch_images, self.keep_prob: 1.0})
        return predictions
