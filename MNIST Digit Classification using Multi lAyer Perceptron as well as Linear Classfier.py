#Using Linear Model

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
tf.reset_default_graph()

batch_size = 50
epochs = 100


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
  

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


X_train_flat = X_train.reshape(50000, -1)
X_val_flat   = X_val.reshape(10000, -1)
X_test_flat  = X_test.reshape(10000, -1)


y_train_categorical = keras.utils.to_categorical(y_train)
y_val_categorical = keras.utils.to_categorical(y_val)
y_test_categorical = keras.utils.to_categorical(y_test)


input_X = tf.placeholder(tf.float32, shape = [None, 784], name = 'input_X')
input_y = tf.placeholder(tf.float32, shape = [None, 10], name = 'input_y')


W = tf.get_variable('W', shape = [784, 10], dtype = tf.float32)
b = tf.get_variable('b', shape = [10], dtype = tf.float32)


logits = input_X @ W + b
probas = tf.nn.softmax(logits)
argmax = tf.argmax(probas, 1)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = input_y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for i in range(epochs):
    
    batch_losses = []
    
    for batch_start in range(0, X_train_flat.shape[0], batch_size): 
        _, batch_loss = sess.run([optimizer, loss], {input_X: X_train_flat[batch_start:batch_start+batch_size], 
                                             input_y: y_train_categorical[batch_start:batch_start+batch_size]})
        batch_losses.append(batch_loss)
    
    train_loss = np.mean(batch_losses)
    val_loss = sess.run(loss, {input_X: X_val_flat, input_y: y_val_categorical})
    
    
    train_score = accuracy_score(y_train, sess.run(argmax, feed_dict = {input_X : X_train_flat}))
    valid_score = accuracy_score(y_val, sess.run(argmax, {input_X: X_val_flat}))
    
    print('Epoch %d Training_score: %f' % (i, train_score), 'Dev_set_score: %f' % valid_score)
    
    
    
    
    
#Using Multi Layer Perceptron
    
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph()
from sklearn.metrics import accuracy_score


batch_size = 50
epochs = 100


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test
  

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


X_train_flat = X_train.reshape(50000, -1)
X_val_flat   = X_val.reshape(10000, -1)
X_test_flat  = X_test.reshape(10000, -1)


y_train_categorical = keras.utils.to_categorical(y_train)
y_val_categorical = keras.utils.to_categorical(y_val)
y_test_categorical = keras.utils.to_categorical(y_test)


input_X = tf.placeholder(tf.float32, shape = [None, 784], name = 'input_X')
input_y = tf.placeholder(tf.float32, shape = [None, 10], name = 'input_y')


W = tf.get_variable('W', shape = [784, 10], dtype = tf.float32)
b = tf.get_variable('b', shape = [10], dtype = tf.float32)


hidden_layer1 = tf.layers.dense(input_X, 512, activation = tf.nn.sigmoid)
hidden_layer2 = tf.layers.dense(hidden_layer1, 256, activation = tf.nn.sigmoid)
hidden_layer3 = tf.layers.dense(hidden_layer2, 128, activation = tf.nn.sigmoid)
outer_layer = tf.layers.dense(hidden_layer3, 10, activation = tf.nn.sigmoid)
probas = tf.nn.softmax(outer_layer)
argmax = tf.argmax(probas, 1)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = outer_layer, labels = input_y)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
#sess = reset_tf_session()
  for i in range(epochs):
    
    batch_losses = []
    
    for batch_start in range(0, X_train_flat.shape[0], batch_size): 
        _, batch_loss = sess.run([optimizer, loss], {input_X: X_train_flat[batch_start:batch_start+batch_size], 
                                             input_y: y_train_categorical[batch_start:batch_start+batch_size]})
        batch_losses.append(batch_loss)
    
    train_loss = np.mean(batch_losses)
    val_loss = sess.run(loss, {input_X: X_val_flat, input_y: y_val_categorical})
    
    
    train_score = accuracy_score(y_train, sess.run(argmax, feed_dict = {input_X : X_train_flat}))
    valid_score = accuracy_score(y_val, sess.run(argmax, {input_X: X_val_flat}))
    
    print('Epoch %d Training_score: %f' % (i, train_score), 'Dev_set_score: %f' % valid_score)
