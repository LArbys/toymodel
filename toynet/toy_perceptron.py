import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers as L
import tensorflow.contrib.slim as slim

def multilayer_perceptron(input_data):
    # Parameters
    learning_rate = 0.0001
    training_epochs = 200000
    batch_size = 100
    display_step = 1000

    # Network Parameters
    n_input = 5 # PID_score data input (five scores)
    n_hidden_1 = 32 # 1st layer number of neurons
    n_hidden_2 = 32 # 2nd layer number of neurons
    n_hidden_3 = 32
    n_classes = 2 # PID Score total classes ([1,0]: 1e1p, [0,1]:ExtBNB)

    # tf Graph input
    #X = tf.placeholder("float", [None, n_input])
    #Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(input_data, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return out_layer



# script unit test
if __name__ == '__main__':
    #x = tf.placeholder(tf.float32, [50,512,512,1])
    net = multilayer_perceptron()
    
