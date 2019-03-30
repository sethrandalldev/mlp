# Multilayer Perceptron Project

The goal of this project is to implement the feed-forward algorithm for applying a multilayer perceptron to a data point and the backpropagation algorithm for training a multilayer perceptron. To simplify this problem, we will assume that this MLP does binary classification.

My project was unable to complete the binary classification. My testing was done on classifying XOR, but my results ended up not separating the 0 or 1 results in any meaningful, consistent manner. I feel that my backpropogation was the reason behind this. I used the perceptron class you provided for us.

My results, for the XOR, ended up with outputs that were within .05 of each other, when the correct answers would need to have been closer to 0 and 1, respectively, so a larger gap between the outputs. I tested the results using 100 iterations as my termination condition, so I feel that my mlp had plenty of iterations to alter the weights enough to properly classify the inputs. I used a learning rate of 0.001.
