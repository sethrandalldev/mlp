import perceptron as perceptron
import math
import numpy as np

LEARNING_RATE = 0.001

def classify(mlp, inputs):
    hidden_layer = mlp[0]
    output_layer = mlp[1]

    out = []
    for input in inputs:
        results = feedforward(input, hidden_layer, output_layer)
        out.append(results[1])

    return out


def train(M, data, targets):

    # Initialize all the weights in all units to random values
    hidden_layer = []
    for m in range(M):
        hidden_layer.append(perceptron.initialize_perceptron(len(data[0]) + 1))
    output_layer = perceptron.initialize_perceptron(M + 1)

    iter = 0
    max_iter = 100

    while iter < max_iter:

        for d in range(len(data)):
            point = data[d]
            target = targets[d]

            # Compute the values for every unit in the network
            hiddens, output = feedforward(point, hidden_layer, output_layer)

            output_delta = output * (1-output) * (target - output)

            hidden_deltas = [hiddens[h] * (1-hiddens[h]) * (output_delta * output_layer.weights[h]) for h in range(len(hiddens))]

            for i in range(len(output_layer.weights)):
                output_layer.weights[i] += LEARNING_RATE * output_delta * output


            # For each hidden unit
            for i in range(len(hidden_layer)):
                # For each weight in the hidden unit
                for j in range(len(hidden_layer[i].weights)):
                    hidden_layer[i].weights[j] += LEARNING_RATE * hidden_deltas[i] * hiddens[i]

        iter += 1

    return (hidden_layer, output_layer)

def feedforward(input, hidden, output):
    hiddens = [hidden[h](input) for h in range(len(hidden))]
    output = output(hiddens)
    return hiddens, output

def sigmoid(t):
    return 1 / (1 + math.exp(-t))


