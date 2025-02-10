"""
Linear Regression Module

This module implements a general multiple linear regression model,
with parameters `w` and bias `b`. Uses batch gradient descent.

Functions:
 - linear_model(X,w,b): computes output of a linear model.
 - compute_cost(X,y,w,b): computes mean squared error cost function.
 - compute_gradients(X,y,w,b): computes derivatives.
 - train_linear_model(X, y, w_init, b_init, max_iter, alpha, epsilon): train linear model using gradient descent
 - get_score(y_true, y_pred): computes coefficient determination of the predicction
 
Author: Alexander Burgos
Fecha: 2025-02-10
"""


import numpy as np

def linear_model(X, w, b):
    '''
    Computes the model for a training example

    Args:
        X (ndarray): Shape (m,n) training examples with n features
        w (ndarray): Shape (,n) n features weights
        b (scalar): Bias parameter
    
    Returns:
        f_wb (ndarray): Shape (m,) predicted outputs
    
    '''
    f_wb = np.dot(X,w) + b

    return f_wb


def compute_cost(X, y, w, b):
    '''
    Computes squared error cost function for the given training set

    Args:
        X (ndarray): Shape (m,n) training examples with n features
        w (ndarray): Shape (n,) n features weights
        y (ndarray): Shape (m,) true outputs
        b (scalar): bias parameter
    
    Returns:
        J_wb (scalar): cost function value
    '''

    # Training examples
    m = X.shape[0]

    loss = linear_model(X, w, b) - y
    J_wb = (1./2.*m) * np.sum(loss**2)

    return J_wb

def compute_gradients(X, y, w, b):
    '''
    Computes gradients for each weight and bias

    Args:
        X (ndarray): Shape (m,n) training examples with n features
        w (ndarray): Shape (n,) n features weights
        y (ndarray): Shape (n,) true outputs
        b (scalar): bias parameter
    
    Returns:
        dJ_dw (ndarray): Shape (n,) weight gradients
        dJ_db (scalar): bias gradient
    '''    
    m = X.shape[0]

    loss = linear_model(X,w,b) - y
    dJ_dw = (1./m) * np.sum(np.reshape(loss, (loss.shape[0],1)) * X, axis=0)
    dJ_db = (1./m) * np.sum(loss)

    return dJ_dw, dJ_db

def train_linear_model(X, y, w_init, b_init, max_iter=1000, alpha=1.e-6, epsilon=1.e-3):
    '''
    Implements gradient descent algorithm to train the linear model
    Args:
        X (ndarray): Shape (m,n) training examples with n features
        y (ndarray): Shape (m,) true outputs
        w_init (ndarray): Shape (n,) n features weights
        b_init (ndarray): Shape (n,) true outputs
        b (scalar): bias parameter
        max_inter (scalar): maximun number of gradient descent steps
        alpha (scalar): learning rate
        epsilon (scalar): defines convergence. Cost function difference between two consecutive interations 
    
    Returns:
        w (ndarray): Shape (n,) optimized features weights
        b (scalar): Optimized bias  
        J_hist (list): History of cost function values 
    '''
    iter = 0

    # Initializating parameters
    w = np.copy(w_init)
    b = np.copy(b_init)

    J_wb = 0.
    J_hist = []
    
    # Repeating gradient descent algorithm until max_iter or convergence 
    while iter <= max_iter:
        iter += 1

        # Gradients
        dJ_dw, dJ_db = compute_gradients(X, y, w, b)
        
        # Update parameters
        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        # Cost function with updated parameters
        J_wb_curr = compute_cost(X, y, w, b)

        # Cost function values over iterations
        J_hist.append(J_wb_curr)

        # Check if convergence achieved
        consecutive_diff = abs(J_wb_curr - J_wb)
        if consecutive_diff <= epsilon:
            print(f"Convergence achieved in {iter} gradient descent steps.")
            break
        
        J_wb = J_wb_curr
    else:
        print(f"Convergence not achieved in {max_iter} gradient descent steps.")    
    
    return w, b, J_hist



def get_score(y_true, y_pred):
    """
    Computes coefficient determination of the predicction
    Own implementation of .score() method from sklearn

    Args:
        y_true (ndarray): Shape (m,) target values
        y_pred (ndarray): Shape (m,) prediction using a linear model

    Returns:
        R_2 (float): score

    """

    u = np.sum((y_true - y_pred)**2)
    v = np.sum((y_true - y_true.mean())**2)

    R_2 = 1. - (u/v)

    return R_2