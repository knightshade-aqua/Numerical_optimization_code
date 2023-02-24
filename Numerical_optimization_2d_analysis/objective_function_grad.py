import numpy as np 

global scale
scale = 1

""" objective_function_grad.py is used to obtain the objective function gradients values. The values are hard-coded """

def rosenbrock(state):
    x = state[0]
    y = state[1]*scale
    # Return the rosenbrock function and its gradient for the given ranges of x and y
    grad = np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])    # gradient of the function
    return grad

def ackley(state):
    x = state[0]
    y = state[1]*scale
    # Return the ackley function and its gradient for the given ranges of x and y
    grad = np.array([-0.4 * x * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) + 0.5 * np.pi * np.sin(2 * np.pi * x) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))), -0.4 * y * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) + 0.5 * np.pi * np.sin(2 * np.pi * y) * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))])    # gradient of the function
    return grad

def booth(state):
    x = state[0]
    y = state[1]*scale
    grad = np.array([2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5), 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)])    # gradient of the function
    return grad

def styblinski_tang_2d(state):
    x = state[0]
    y = state[1]
    grad = np.array([x**3 - 16 * x + 5, y**3 - 16 * y + 5])    # gradient of the function
    return grad