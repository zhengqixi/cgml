import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os
class poopoonet:

    def __init__(self):
        self.sess = tf.Session()

    def build_model(self, input_shape, num_output):
        #shape: [dimension1, dimension2, channels]
        #Too hard to do minibatch everything gonna be online cuz that's superior anyway
        self.inputs = tf.placeholder(tf.float32, shape=[1, input_shape[0], input_shape[1], input_shape[2]], name='input')
        self.num_output = num_output
        self.input_shape = input_shape
        with slim.arg_scope([slim.layers.convolution, slim.layers.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.5)):
            net = slim.layers.convolution(self.inputs, num_outputs=50, kernel_size=3, scope='conv1')
            net = slim.layers.max_pool2d(net, 3, scope='pool1')
            net = slim.layers.convolution(net, num_outputs=30, kernel_size= 2, stride=1,scope='conv2')
            net = slim.layers.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.layers.convolution(self.inputs, num_outputs=15, kernel_size=3, scope='conv3')
            net = slim.layers.flatten(net, scope='flat')
            net = slim.layers.fully_connected(net, 100)
            net = slim.layers.fully_connected(net, 50)
            net = slim.layers.fully_connected(net, num_output, activation_fn=None, weights_regularizer=None)
            self.net = net

    def train_model(self, learning_rate, train_data, validation_data, validation_epoch, outdir=None):
        self.labels = tf.placeholder(tf.float32, shape=[1, self.num_output])
        predict_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.net, labels=self.labels)
        weight_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        total_loss = predict_loss + weight_loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.trainer = slim.learning.create_train_op(total_loss, optimizer)
        self.sess.run(tf.global_variables_initializer())
        batch_image, batch_label = self.get_batch(train_data)
        loss = 0
        epoch = 1
        losses_vector = []
        epoch_vector = []
        validation_accuracy = []
        validation_epoch_time= []
        for image, label in zip(batch_image, batch_label):
            _, loss = self.sess.run([self.trainer, total_loss], feed_dict={self.inputs:image, self.labels:label})
            losses_vector.append(float(loss))
            epoch_vector.append(epoch)
            if epoch%validation_epoch == 0:
                predictions = self.infer_model(validation_data)
                validation_accuracy.append(self.accuracy(predictions, validation_data['labels']))
                validation_epoch_time.append(epoch)
            epoch += 1

        return epoch_vector, losses_vector, validation_epoch_time, validation_accuracy

    def get_batch(self, data):
        batch_image = []
        batch_label = []
        for image, label in zip(data['images'], data['labels']):
            batch_image.append([np.reshape(image, (self.input_shape[0], self.input_shape[1], self.input_shape[2]))])
            batch_label.append([np.reshape(label, self.num_output)])

        return batch_image, batch_label

    def accuracy(self, predictions, labels):
        num_correct = 0
        num_total = len(labels)
        for prediction, label in zip(predictions, labels):
            if np.argmax(prediction) == np.argmax(label):
                num_correct += 1
        return num_correct/num_total

    def infer_model(self, data):
        batch_images, batch_labels = self.get_batch(data)
        predictions = []
        for image, label in zip(batch_images, batch_labels):
             predictions.append(self.sess.run(tf.nn.softmax(logits=self.net), feed_dict={self.inputs:image})[0])

        return predictions
