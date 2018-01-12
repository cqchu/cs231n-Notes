import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #计算num_train个case的loss并累加
  for i in range(num_train):           # 对于每个训练数据
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]  # 即classifier在正确分类上的得分
    for j in range(num_classes):       # 每次迭代更新 dW 一列
      if j == y[i]:                    # 此次不用更新
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:                   # dW需要更新
        loss += margin
        dW[:, j] += X[i, :].T                 
        dW[:, y[i]] -= X[i, :].T       # 不要忘记那个减去的y[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  Y = X.dot(W)   # N*C, 每一行是一个case的各项得分
  y_score = Y[np.arange(Y.shape[0]), y]
  y_score = np.reshape(y_score, (Y.shape[0] ,1))
  Y = Y - y_score + 1
  Y[np.arange(Y.shape[0]), y] -= 1
  Y[Y<0] = 0
  loss = np.sum(Y)/Y.shape[0]
  loss = loss + reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  Y[Y>0] = 1
  row_sum = np.sum(Y, axis = 1)  # 用于计算是否需要减去那个y[i]
  Y[np.arange(Y.shape[0]), y] = -row_sum
  dW = np.dot(X.T, Y)/Y.shape[0] + 2 * reg * W
  return loss, dW
