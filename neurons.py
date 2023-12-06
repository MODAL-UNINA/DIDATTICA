import numpy as np

# --- example 1 ---
# given the inputs.
inputs = np.array([0.2, 0.4, 0.6])

# define weights and bias.
weights = np.array([0.5, 0.3, 0.2])
bias = 0.8

# define a nonlinear activation function.
def activation_function(z):
    a =  1 / (1 + np.exp(-z))
    return a

# get the output of the neuron, np.dot() calculates the dot product of two arrays.
z = np.dot(inputs, weights.T) + bias
output = activation_function(z)

print('output:', output)


# --- example 2 ---
# given the inputs.
inputs = np.array([[0.2, 0.4, 0.6],
                   [0.3, 0.5, 0.7],
                   [0.4, 0.6, 0.8]])

# define weights and bias.
weights = np.array([[0.5, 0.3, 0.2],
                    [0.4, 0.6, 0.5],
                    [0.7, 0.5, 0.3]])

bias = np.array([0.8, 0.2, 0.1])

# define a nonlinear activation function.
def activation_function(z):
    a =  1 / (1 + np.exp(-z))
    return a

# get the output of neurons, np.dot() calculates the dot product of two arrays.
z = np.dot(inputs, weights.T) + bias
output = activation_function(z)

print('output:', output)


# --- example 3 ---
# given the inputs.
inputs = np.array([[[0.18, 0.21, 0.64],
                    [0.40, 0.45, 0.08],
                    [0.69, 0.26, 0.28]],
                   [[0.63, 0.78, 0.45],
                    [0.60, 0.40, 0.86],
                    [0.25, 0.45, 0.26]],
                   [[0.39, 0.94, 0.36],
                    [0.64, 0.92, 0.82],
                    [0.41, 0.91, 0.59]]])

# define weights and bias, with random values.
np.random.seed(0)
weights = np.random.rand(3, 3, 3)
print('weights:', weights)

bias = np.random.rand(3)
print('bias:', bias)

def activation_function(z):
    a =  1 / (1 + np.exp(-z))
    return a

z = np.dot(inputs, weights.T) + bias

# calculate the output of neurons.
output = activation_function(z)

print('output:', output)
