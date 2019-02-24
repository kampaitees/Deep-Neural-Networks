#Importing Dependencies

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)


#Computing the following loss using tensorflow
#Loss(y_hat, y) = np.power((y_hat - y),2)

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss

#Output = 9    
    
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)

#Output
#Tensor("Mul:0", shape=(), dtype=int32)

print(sess.run(c))

#Output = 20

x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

#Ouptut = 6



#X = tf.constant(np.random.randn(3,1), name = "X")

#You might find the following functions helpful:

#    tf.matmul(..., ...) to do a matrix multiplication
#    tf.add(..., ...) to do an addition
#    np.random.randn(...) to initialize randomly


def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    
    
    X = tf.constant(np.random.randn(3, 1), name = 'x')
    W = tf.constant(np.random.randn(4, 3), name = 'W')
    b = tf.constant(np.random.randn(4, 1), name = 'b')
    Y = tf.add(tf.matmul(W, X), b
    
   
    sess = tf.Session()
    result = sess.run(Y)
    
    
    # close the session 
    sess.close()

    return result

print( "result = " + str(linear_function()))

#Output := result 	[[-2.15657382] [ 2.95891446] [-1.08926781] [-0.84538042]]
               

#Note that there are two typical ways to create and use sessions in tensorflow:

#Method 1:

#sess = tf.Session()
# Run the variables initialization (if needed), run the operations
#result = sess.run(..., feed_dict = {...})
#sess.close() # Close the session

#Method 2:

#with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
    # result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)


def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = 'x')

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

   
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x : z})
   
    return result
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

#Output 
sigmoid(0) = 0.5
sigmoid(12) = 0.999994


    
#1.3 - Computing the Cost

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
   
    
    # Creating the placeholders for "logits" (z) and "labels" (y) 
    z = tf.placeholder(tf.float32, name = 'z')
    y = tf.placeholder(tf.float32,   name = 'x')
    
    # Using the loss function 
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    
    # Creating a session 
    sess = tf.Session()
    
    # Running the session 
    cost = sess.run(cost, feed_dict = {z : logits, y : labels})
    
    # Closing the session (approx. 1 line). See method 1 above.
    sess.close()
    
    
    return cost    
    
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


#Output
cost 	 = [ 1.00538719 1.03664088 0.41385433 0.39956614] 



#1.4 - Using One Hot encodings

#Many times in deep learning you will have a y vector with numbers ranging from 0 to C-1, where C is the 
#number of classes. If C is for example 4, then you might have the following y vector which you will need to convert as follows:

#This is called a "one hot" encoding, because in the converted representation exactly one element of each column is
#"hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. 
#In tensorflow, you can use one line of code:

    #tf.one_hot(labels, depth, axis)

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
   
    
    # Creating a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, tf.int32, name = 'C')
    
    # Using tf.one_hot, be careful with the axis 
    one_hot_matrix = tf.one_hot(labels,C, axis = 0)
    
    # Creating the session 
    sess = tf.Session()
    
    # Running the session 
    one_hot = sess.run(one_hot_matrix)
    
    # Closing the session (approx. 1 line). See method 1 above.
    sess.close()
    
   
    return one_hot


labels = np.array([1,2,3,2,1])
one_hot = one_hot_matrix(labels, C = 3)
print ("one_hot = " + str(one_hot))


#Output
one_hot = [[ 0.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  1.]
 [ 0.  1.  0.  1.  0.]]
 
 
 
 
 #1.5 - Initialize with zeros and ones
def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
   
    
    # Creating "ones" tensor using tf.ones(...). 
    ones = tf.ones(shape)
    
    # Creating the session 
    sess = tf.Session()
    
    # Running the session to compute 'ones' 
    ones = sess.run(ones)
    
    # Closing the session 
    sess.close()
    
    return ones

print ("ones = " + str(ones([3])))

#Output
ones = [ 1.  1.  1.]



#2 - Building your first neural network in tensorflow

 #   Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
 #   Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

#Note that this is a subset of the SIGNS dataset. The complete dataset contains many more signs.
 
#Here are examples for each number, and how an explanation of how we represent the labels. These are the original 
#pictures, before we lowered the image resolutoion to 64 by 64 pixels. 

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Change the index below and run the cell to visualize some examples in the dataset.

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


#Output y = 5


#As usual you flatten the image dataset, then normalize it by dividing by 255. On top of that, you will convert each label to a 
#one-hot vector as shown in Figure 1. Run the cell below to do so.



# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))



#Output
number of training examples = 1080
number of test examples = 120
X_train shape: (12288, 1080)
Y_train shape: (6, 1080)
X_test shape: (12288, 120)
Y_test shape: (6, 120)


#Note that 12288 comes from 64×64×364×64×3. Each image is square, 64 by 64 pixels, and 3 is for 
#the RGB colors. Please make sure all these shapes make sense to you before continuing.

#The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX. The SIGMOID output layer has been 
#converted to a SOFTMAX. A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.


#2.1 - Create placeholders

#Your first task is to create placeholders for X and Y. This will allow you to later pass your training
#data in when you run your session.

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

   
    X = tf.placeholder(tf.float32,[n_x,None],name = "X")
    Y = tf.placeholder(tf.float32,[n_y,None],name = "Y")
    
    
    return X, Y
    
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


#Output
X = Tensor("X:0", shape=(12288, ?), dtype=float32)
Y = Tensor("Y:0", shape=(6, ?), dtype=float32)



#2.2 - Initializing the parameters
#W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
 
    W1 = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1], initializer = tf.zeros_initializer())
   


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
    
    
  tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
#Output
W1 = <tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>
b1 = <tf.Variable 'b1:0' shape=(25, 1) dtype=float32_ref>
W2 = <tf.Variable 'W2:0' shape=(12, 25) dtype=float32_ref>
b2 = <tf.Variable 'b2:0' shape=(12, 1) dtype=float32_ref>
#As expected, the parameters haven't been evaluated yet.


#2.3 - Forward propagation in tensorflow

 #   tf.add(...,...) to do an addition
 #   tf.matmul(...,...) to do a matrix multiplication
 #   tf.nn.relu(...) to apply the ReLU activation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
                                                                        # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                     # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                                 # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                    # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                                 # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                    # Z3 = np.dot(W3,Z2) + b3
   
    
    return Z3

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
    
    
#Output
Z3 = Tensor("Add_2:0", shape=(6, ?), dtype=float32)


#2.4 Compute cost
 def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  
    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
    
    
 #Output
 cost = Tensor("Mean:0", shape=(), dtype=float32)

    
#2.5 - Backward propagation & parameter updates    
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)


    # Initialize parameters
   
    parameters = initialize_parameters()

    
    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters)
 
    
    # Cost function: Add cost function to tensorflow graph
   
    cost = compute_cost(Z3, Y)
   
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
             
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
              
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)

Cost after epoch 0: 1.855702
Cost after epoch 100: 1.016458
Cost after epoch 200: 0.733102
Cost after epoch 300: 0.572940
Cost after epoch 400: 0.468774
Cost after epoch 500: 0.381021
Cost after epoch 600: 0.313822
Cost after epoch 700: 0.254158
Cost after epoch 800: 0.203829
Cost after epoch 900: 0.166421
Cost after epoch 1000: 0.141486
Cost after epoch 1100: 0.107580
Cost after epoch 1200: 0.086270
Cost after epoch 1300: 0.059371
Cost after epoch 1400: 0.052228


Parameters have been trained!
Train Accuracy: 0.999074
Test Accuracy: 0.716667
