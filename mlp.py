import perceptron
import math
import numpy as np

learning_rate = 0.001

def classify(mlp, inputs):
    input_w = mlp[0]
    hidden_w = mlp[1]
    bias_w = mlp[2]
    outputs = []

    for i in range(len(inputs)):
        hidden = [0 for x in range(len(hidden_w))]
        for h in range(len(hidden_w)):
            for j in range(len(input_w)):
                # Possibly wrong input weight
                hidden[h] += inputs[i][j] * input_w[i][j]
            hidden[h] = sigmoid(hidden[h])

        output = 0
        for h in range(len(hidden)):
            output += hidden[h] * hidden_w[h]
        output = sigmoid(output)
        outputs.append(output)

    return outputs

def train(M, data, targets):
    # Setup up the MLP by initializing the input layer and hidden layer with random value weights
    data_length = len(data[0])

    bias = [np.random.uniform(), np.random.uniform()]

    # Initialize the hidden layer values to 0
    Z = [0 for x in range(M)]
    Z_weights = [np.random.uniform() for x in range(M)]

    # Initialize the input layer weights for each perceptron related to a hidden unit
    W = []
    for i in range(data_length):
        toAppend = []
        for m in range(M):
            toAppend.append(np.random.uniform())
        W.append(toAppend)

    # Condition: 100 iterations
    iter = 0
    max_iter = 10

    while iter < max_iter:

        outputs = []
        for d in range(len(data)):
            # Calculate the values of the hidden units in the hidden layer
            for i in range(M):
                Z[i] += bias[0]
                for j in range(data_length):
                    Z[i] += data[d][j] * W[j][i]
                Z[i] = sigmoid(Z[i])

            # Calculate the value of the output
            out = bias[1]
            for i in range(len(Z)):
                out += Z[i] + Z_weights[i]
            out = sigmoid(out)

            # Backprop
            output_delta = out * (1 - out) * (targets[d] - out)

            for z in range(len(Z_weights)):
                Z_weights[z] += output_delta*learning_rate*Z[z]

            hidden_deltas = [Z[z]*(1-Z[z])*Z_weights[z] for z in range(len(Z))]

            for i in range(data_length):
                for j in range(len(Z)):
                    W[j][i] += learning_rate * hidden_deltas[j] * data[d][j]

            outputs.append(out)

        iter += 1

    return W, Z_weights, bias


def sigmoid(t):
    return 1 / (1 + math.exp(-t))


