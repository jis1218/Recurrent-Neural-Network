# coding: utf-8
'''
Created on 2018. 5. 14.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    
    
    
    
    idx2char = ['h', 'i', 'e', 'l', 'o']
    x_data = [[0, 1, 0, 2, 3, 3]] #hihell
    x_one_hot = [[[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0]]]
    
    y_data = [[1, 0, 2, 3, 3, 4]] #ihello
    
    sequence_length = len(x_data[0])
    hidden_size = len(idx2char)
    batch_size = 1
    
    x = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])
    y = tf.placeholder(tf.int32, [None, sequence_length])
    
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    output, _states = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
    weights = tf.ones([batch_size, sequence_length])
    
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=output, targets=y, weights=weights)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    
    prediction=tf.argmax(output, axis=2)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            l, _ = sess.run([loss, train], feed_dict={x:x_one_hot, y:y_data})
            result = sess.run(prediction, feed_dict={x:x_one_hot})
            print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)
            
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction str: ", ''.join(result_str))
    
    
    
    pass