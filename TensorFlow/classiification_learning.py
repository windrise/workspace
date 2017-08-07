# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


#define placeholder for inputs to network


#add output layer



#the error between prediction and real data



sess = tf.Session()
#important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
