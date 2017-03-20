#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 20:20:55 2017
http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
@author: ros-devel
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Data Generation

def gen_data(size=10):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)
  
  
 # adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
      print("generating epoch %i"%(i))
      yield gen_batch(gen_data(), batch_size, num_steps)
        
#%% RNN Cell creation
def rnn_cell(W,b,rnn_input,state):
  # W -> [(input_size + hidden_layer_output_size) , hidden_layer_output_size]
  # b -> [ hidden_layer_output_size ]
  input_and_state = tf.concat([rnn_input, state], axis=1) # concatenating [batch_size x (input_size + hidden_layer_output_size) ]
  out = tf.tanh(tf.matmul(input_and_state, W) + b)
  print("RNN Cell Output Size %s"%(str(out.get_shape())))
  
  return out
  

  
  
        
#%% RNN inputs
num_steps = 5 # number of states retained  
batch_size = 50
num_classes = 2
state_size = 4 # outputs of the hidden layer so the state size is 4 x 1 
learning_rate = 0.1

x = tf.placeholder(shape= [batch_size,num_steps],dtype=tf.int32,name="input_placeholder")
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros(shape=[batch_size,state_size])

# Inputs to the network
# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x_one_hot = tf.one_hot(x,num_classes)
rnn_inputs = tf.unstack(x_one_hot,axis=1)

#%% RNN Hiddeen layer Weighs and biases
weight_sizes = [(num_classes + state_size,state_size), (state_size,num_classes)]
bias_sizes = [(state_size),(num_classes)]

weights =[]
biases = []
for i,j in enumerate(weight_sizes):
  weights.append(tf.Variable(initial_value=tf.random_normal(shape=weight_sizes[i],stddev=0.05)))
  biases.append(tf.Variable(initial_value=tf.zeros(shape=bias_sizes[i])))

#W = tf.Variable(initial_value=tf.random_normal(shape=[num_classes + state_size,state_size],stddev=0.05))
#b = tf.Variable(initial_value=tf.zeros(shape=[state_size]))

#%%  RNN hidden layer output 
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
  state = rnn_cell(weights[0],biases[0],rnn_input,state)
  rnn_outputs.append(state)
  
final_state = rnn_outputs[-1]

#%% Output layer
y_as_list = tf.unstack(y,num=num_steps,axis=1)
logits = [tf.matmul(rnn_output,weights[1]) + biases[1] for rnn_output in rnn_outputs]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit) for logit,label in zip(logits,y_as_list)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)


#%% Train the network


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses
  
#%% Plot results
training_losses = train_network(1,num_steps)
plt.plot(training_losses)

  







