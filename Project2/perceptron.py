import numpy as np

def perceptron_train(X, Y):
    #Get the number of samples and the number of features
    numSamples, numFeatures = X.shape

    #Initialize b and w as 0
    b = 0
    w = np.zeros(numFeatures)

    #Set updated to True so loop begins
    updated = True
    while updated == True:
        #Set updated to False
        updated = False
        #Traverse through all the samples
        for i in range(numSamples):
            #Set activation to 0
            activation = 0
            #Traverse through all the features
            for j in range(numFeatures):
                #Calculate a for the sample dimension and add it to activation
                activation += w[j]*X[i, j] + b

            #Calculate y*a
            checkVar = activation*Y[i]
            #Check if the product is less than 0
            if checkVar <= 0:
                #If the product is less than 0, then traverse through the features again
                for j in range(numFeatures):
                    #Assign new values to w
                    w[j] = w[j] + Y[i] * X[i, j]
                #Assign new value to b
                b = b + Y[i]
                #Set updated to True so loop continues
                updated = True

    return w,b
    
def perceptron_test(X, Y, w, b):
    #Get the number of samples and the number of features
    numSamples, numFeatures = X.shape
    
    #Set an empty precition array
    yPred = []
    #Traverse through all the samples
    for i in range(numSamples):
        #Set activation to 0
        activation = 0
        #Traverse through the sample features
        for j in range(numFeatures):
            #Calculate the activation
            activation += w[j]*X[i, j] + b
        
        #If activation is negative, predict the sample as -1
        if activation <= 0:
            yPred.append(-1)
        #Else predict it as postive
        else:
            yPred.append(1)

    #Set the count of correctly predicted samples to 0
    count = 0
    #Traverse the samples
    for i in range(numSamples):
        #Check if the prediction matches the actual value
        if (Y[i] == yPred[i]):
            #If it does, increase count by 1
            count += 1
    
    #Calculate accuracy by dividing count by number of samples
    accuracy = count/numSamples

    return accuracy