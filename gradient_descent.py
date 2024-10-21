import numpy as np

def gradient_descent(gradient, x_init, lrate):
    #While the magnitude of the gradient is greater than 0.0001
    while (np.linalg.norm(gradient(x_init)) > 0.0001):
        #Update x_init with the gradient descent equation
        x_init = x_init - lrate * gradient(x_init)
    return x_init