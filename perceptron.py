import numpy as np

# The "logistic" function, often called "sigmoid"
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

# A class that represents a single perceptron
class Perceptron :
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation
    def dimension(self) :
        return len(self.weights)-1
    def __call__(self, inputs) :
        return self.activation(np.dot(self.weights, inputs + [1]))
    def __str__(self) :
        return ",".join([str(w) for w in self.weights])

from random import uniform

def initialize_perceptron(n) :
    return Perceptron([np.random.uniform() for n in range(n)], sigmoid)

