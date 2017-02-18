import tensorflow as tf
import numpy as np

class spiral_model:
    def __init__(self, session, layers, rate, iterations, gamma):
        self.session = session
        self.layers = layers
        self.rate = rate
        self.iterations = iterations 
        self.gamma = gamma
        self.build()

    def add_variable(self, name, shape):
        var = tf.get_variable(name=name,
                              dtype=tf.float32,
                              shape=shape,
                              initializer=tf.random_normal_initializer())
        tf.add_to_collection('model_vars', var)
        tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
        return var

    def build(self):
        self.x = tf.placeholder(tf.float32, shape=[1, self.layers[0]])
        self.y = tf.placeholder(tf.float32, shape=[1, self.layers[0]])
        self.c = tf.placeholder(tf.float32, shape=[])

        self.weights_x = []
        self.weights_y = self.add_variable('w_y', [self.layers[0], self.layers[1]]) 
        self.biases = []

        for ii in range(0, len(self.layers)-1):
            self.weights_x.append(self.add_variable('w_x' + str(ii), [self.layers[ii], self.layers[ii+1]]))
            self.biases.append(self.add_variable('b' + str(ii), [1, self.layers[ii+1]]))

        chat = tf.sigmoid(tf.matmul(self.x, self.weights_x[0]) + tf.matmul(self.y, self.weights_y) + self.biases[0])

        for ii in range(1, len(self.layers)-1):
            chat = tf.sigmoid(tf.matmul(chat, self.weights_x[ii]) + self.biases[ii])

        self.chat = tf.nn.softmax(chat)

        cost = tf.reduce_mean(0.5*(self.chat-self.c)**2)
        l2 = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = cost + self.gamma*l2

    def initialize(self):
        variables = tf.get_collection('model_vars')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.rate).minimize(self.loss, var_list=variables)
        self.session.run(tf.global_variables_initializer())

    def train(self, x_data, y_data, class_data):
        for step in range(self.iterations):
            for x, y , c in zip(x_data, y_data, class_data):
                self.session.run(self.optimizer, feed_dict={self.x: np.ones([1,self.layers[0]])*x, self.y: np.ones([1,self.layers[0]])*y, self.c: c}) 

    def predict(self, x_point, y_point):
        return max(self.session.run(self.chat, feed_dict={self.x: np.ones([1,self.layers[0]])*x_point, self.y: np.ones([1,self.layers[0]])*y_point})[0]) 


if __name__ == "__main__":
    test = spiral_model(tf.Session(), [3, 4, 5], 0.4, 10, 3) 
    test.initialize()
    test.train(range(5), range(5), range(5))
    print(test.predict(4, 6))
    
