from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    z1 = X.dot(W1) + b1
    a1 = np.where(z1 > 0, z1, np.zeros(z1.shape)) # ReLU
    z2 = a1.dot(W2) + b2
    scores = z2 # is not softmax
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # softmax
    z2_exp = np.exp(z2)
    z2_exp_sum = np.reshape(np.sum(z2_exp, axis = 1), (z2_exp.shape[0], 1))
    scores = z2_exp / z2_exp_sum
    y_score = scores[np.arange(scores.shape[0]), y]
    y_score = np.reshape(y_score, (y_score.shape[0], 1))
    loss = np.sum(-np.log(y_score))/y_score.shape[0] + reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    
    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
	# 一些临时变量
    y_exp = np.reshape(z2_exp[np.arange(z2_exp.shape[0]), y], (z2_exp.shape[0], 1))  # exp_z2的第y列的值
    softmax_mask = np.where(scores == y_score, np.ones(scores.shape), np.zeros(scores.shape))
    ReLU_mask = np.where(z1>0, np.ones(z1.shape), np.zeros(z1.shape))
    # compute grad without reg
    dL_dyscore = (-1/N) * 1/y_score                                                                       # softmax's loss funtion
	# 下面这种注释后的算法实际上是工整的链式法则得到的算法，但其生成的高维数组很多数据和mask一乘以后全变成0，之前的计算结果都浪费了---低效
    # dL_dscore = dL_dyscore * np.where(scores == y_score, np.ones(scores.shape), np.zeros(scores.shape))   # dL_dyscore * dyscore_dscore
    # dL_dexpZ2 = dL_dscore * dscore_dexpZ2
	# 此处决定直接计算dyscore_dexpZ2，即y_score所对应的那一列计算梯度
    dL_dexpZ2 = dL_dyscore * ((softmax_mask*z2_exp_sum - y_exp)/(z2_exp_sum**2))  # dL_dyscore * dyscore_dexpZ2
    dL_dZ2 = dL_dexpZ2 * z2_exp
    dL_da1 = dL_dZ2.dot(W2.T)
    dL_dZ1 = dL_da1 * ReLU_mask
	
	# Z2 = a1 * W2 + b2
    grads['W2'] = dL_dW2 = a1.T.dot(dL_dZ2) + reg * 2 * W2
    grads['b2'] = dL_db2 = np.sum(dL_dZ2*1, axis=0)
    # Z1 = X * W1 + b1
    grads['W1'] = dL_dW1 = X.T.dot(dL_dZ1) + reg * 2 * W1
    grads['b1'] = dL_db1 = np.sum(dL_dZ1*1, axis=0)
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      mask = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[mask]
      y_batch = y[mask]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
	  
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    z1 = X.dot(self.params['W1']) + self.params['b1']
    a1 = np.where(z1 > 0, z1, np.zeros(z1.shape)) # ReLU
    z2 = a1.dot(self.params['W2']) + self.params['b2']
    z2_exp = np.exp(z2)
    z2_exp_sum = np.reshape(np.sum(z2_exp, axis = 1), (z2_exp.shape[0], 1))
    scores = z2_exp / z2_exp_sum
    y_pred = np.argmax(scores, axis = 1)

    return y_pred


