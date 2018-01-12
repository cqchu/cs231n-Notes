import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    tmp_score = np.exp(X[i].dot(W))
    score = tmp_score/np.sum(tmp_score)
    loss +=  -np.log(score[y[i]])
	
    for j in range(W.shape[1]):
      if j==y[i]:
        dW[:, y[i]] -= X[i].T * (np.sum(tmp_score)-tmp_score[y[i]])/np.sum(tmp_score)
      else:
        dW[:, j] += X[i].T * tmp_score[j] / np.sum(tmp_score)
  loss = loss / X.shape[0] + 0.5*reg*np.sum(W*W)
  dW = dW / X.shape[0] + reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  tmp = np.exp(X.dot(W))
  sum = np.sum(tmp, axis = 1)
  sum = np.reshape(sum, (sum.shape[0], 1))
  score = tmp/sum;
  loss = np.sum(-np.log(score[np.arange(score.shape[0]), y]))/X.shape[0]
  loss = loss + 0.5*reg*np.sum(W*W)
  
  sum_mask = np.zeros((X.shape[0], W.shape[1]))
  sum_mask[np.arange(X.shape[0]), y] = np.reshape(sum, (sum.shape[0], ))
  dW = X.T.dot(-(sum_mask - tmp)/sum)
  dW = dW / X.shape[0] + reg*W
  
  return loss, dW

