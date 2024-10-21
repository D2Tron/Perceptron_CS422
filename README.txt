Jayam Sutariya
CS 422
Project 2

1. Perceptron
perceptron_train(X, Y): This function takes in the training data and the corresponding labels. It sets b = 0 and w as 
an array of zeros. While the algorithm has not converged, keep traversing through all the training samples. While
traversing through samples, calculate the activation variable using the training data, weights, and bias for each
sample. Calculate the y*a and check if it is less than or equal to 0. If that is the case, update w and b with the 
equations. The loop will stop once the algorithm has converged, and the weights and bias will be returned.

perceptron_test(X, Y, w, b): This function takes in the testing data and the corresponding labels in addition to the 
weights and bias calcualted in the training function. An empty array is created for the predicted labels. Similar to
the training function, the testing samples are traversed. The activation is calculated for each sample. If the 
activation is less than or equal to 0, the sample is predicted as -1. Otherwise, it is predicted as 1. All the samples
are traversed through one more time. This time, it is checked if the prediction matches the actual test data labels. 
For all correctly predicted samples, a count is increased by 1. After the loop, the count is divided by the total 
number of samples to acquire the accuracy, which is then returned.


2. Gradient Descent
gradient_descent(gradient, x_init, lrate): This function takes in the gradient function, the initial value of x as an 
array, and the learning rate. The value of x is modified with the gradient descent equation, until the gradient of the
magnitude of the x value is greater than 0.0001. After that, the x value is returned.