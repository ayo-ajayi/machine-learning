import numpy as np

class Layer_Dense:

    # layer init
    def __init__(self, n_inputs, n_neurons):
        self.weights=0.01*np.random.randn(n_inputs, n_neurons) 
        # could have been np.random.randn(n_neurons, n_inputs), but we avoid the need for transpose all the time
        self.biases=np.zeros((1,n_neurons))
    
    #forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) +self.biases


class Activation_ReLU:
    #forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs): #inputs - output of the model
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True)) #prevent overflow tendencies from: np.exp(inputs)
        probabilities = exp_values/np.sum(exp_values, axis = 1, keepdims=True) 
        self.output = probabilities
        
class Accuracy:
    def calculate(self, output,y):
        predictions=np.argmax(output, axis=1)
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        accuracy=np.mean(predictions==y) 
        return accuracy
        
class Loss:
    # output:out from model, y:intended target values
    def calculate(self,output, y) :
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    #y_pred: from neural network
    #y_true: traget training values
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

       #use one-hot encoded vectors     
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        