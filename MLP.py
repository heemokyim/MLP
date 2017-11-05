"""
PROGRAM INFORMATION: MLP (Multi-layer Perceptron)

Authors:        Viyaleta Peterson
                Muhammad Wasim Ud Din
                Karan Manoj Bhosaale

Institution:    University of Bridgeport
                CPCS 552 - Data Mining
                Pr. Jeongkyu Lee
                Fall 2017

Description:    This program is a library of objects and methods which allows a
                user to implement a multilayer perceptron (Artificial Neural Network).
                There are two classes: Layer and Network. Layer stores information
                with regard to nodes within each layer. Network stores the layer
                information as well as provides various methods, such as train,
                predict, save, load, show, and visualize to the user.
                Private methods in the Network class include initialization,
                forward propagation, and backward propagation and are referenced
                in the training and testing methods.

Dependencies:   time: to track algorithm performance,
                csv: to save predicted values as a .csv file
                pickle: to save and load network parameters
                numpy: for all matrix compuations and various functions and constants
                    NOTE: This implementation requires numpy+mkl wheel
                math: for mathematical constancts
                random: for random number generation
                PIL: for creating a graph of the network as png file
                matplotlib: for plotting

Instructions:   1. Save this file to a particular location
                2. Create a new .py file
                3. In the new file, enter 'from SimpleMLP import Layer, Network'
                4. Set up the network by specifying architecture as a list of Layers
                        with number of nodes and activation function
                        Example: 'net = Network(architecture=[Layer(10), Layer(2, 'relu')], l_rate=0.1)'
                5. Initialize the network with random weight values.
                        Example: 'net.initialize()'
                6. Train the network.
                        Example: 'net.train(X=input_data, y=output_data, epochs=100)'
                7. Save the network architecture.
                        Example: 'net.save('file_name')'
                8. Load the network architecture.
                        Example: 'net.load('file_name')'
                9. Visualize the network in as a .png file.
                        Example: 'net.visualize()'
                10. Look inside the network.
                        Example: 'net.show()'
                11. Use the network to predict the values.
                        Example: 'net.predict(testing_data, verbose=True, save_file='my_results')'



"""

import time
import csv
import pickle
import numpy as np
from math import e
from random import random, seed, uniform
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
plt.style.use('ggplot')  #matplotlib style


class Layer:
    def __init__(self, n, fn='linear', weights=[], bias=0):
        """
        Container for multilayer perceptron hidden layer.

        Args:
            n: number of neurons in the hidden layer
            fn: activation function to be used to calculate values in the layer
            weights: weights from current layer's nodes to next layer's nodes
            bias: layer bias

        Note:
            weights is a list where each item i is a list of wegiths
            corresponding to nodes j in the following layer.
            example: [[1->1, 1->2, 1->3], [2->1, 2->2, 2->3]]
            The first value is the current layer's node id, the second value
            is the following layer's node id.
        """

        self.n = n
        self.fn = fn
        self.bias = bias
        self.weights = weights
        self.values = []            #stores values at nodes of the layer
        self.d = []             #stores partial derivates at each node in backpropagation
        self.weight_deltas = []     #stores weight adjustments in backpropagation


    def initialize(self, n_next_layer, output):
        """
        Initializes a layer in the network with random bias and weights.

        Args:
            n_next_layer: number of nodes in the next layer
            output: boolean indicator whether working with output layer
        """

        self.values = [0] * self.n  #initializes number of nodes
        self.d = [0] * self.n       #initializes partial derivative values for each node

        # For all layers except output, initializes weights and bias to random()
        if output == False:
            self.weights = [
                [uniform(0, 1) for k in range(n_next_layer)]
                    for j in range(self.n)]
            self.bias = uniform(0, 1)


class Network:
    def __init__(self, architecture=[], l_rate=0.1, use_seed=False):
        """
        Container for the network, including layers with node values, weights, and
        bias information, and methods to forward and backward propagate,
        train, test, save and load the network. Also includes visualizations.

        Args:
            architecture: array of Layer instances
            l_rate: learning rate
            use_seed: ???
        """
        #TODO move the learning rate to training method only
        #TODO fix the use_seed parameter, make sure it works

        self.architecture = architecture
        self.n_layers = len(architecture)   #number of layers
        self.l_rate = l_rate
        if use_seed == True: seed(1)        #FIXME


    def initialize(self):
        """
        Method for initialization of the network. Modifies each layer in the network
        by calling the initialize method in the Layer class.
        """

        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                #initializes the output layer with 0 as n_next_layer, flags as output layer
                self.architecture[i].initialize(0, output=True)
            else:
                #initializes the layer with weights leading to the next layer
                n_next_layer = self.architecture[i+1].n
                self.architecture[i].initialize(n_next_layer, output=False)


    def forward_propagate(self):
        """
        Performs forward propagation on every layer in the nextwork.
        Calculates the node value at current layer by applying a user specified
        layer-specific activation function to the sum of dot products between
        previous layer's values and weights from those values to the current node.

        Returns:
            self.architecture[-1].values: calculated values at the output layer
        """
        # Holds activation functions as lambda functions in a dictionary
        activate = {
            'linear':   (lambda x: x),
            'neglin':   (lambda x: -x),
            'inv':      (lambda x: 1/x),
            'squared':  (lambda x: x**2),
            'cube':     (lambda x: x**3),
            'sigmoid':  (lambda x: 1/(1 + e**(-x))),
            'relu':     (lambda x: x if x > 0 else 0),
            'tanh':     (lambda x: np.tanh(x)),
            'leaky':    (lambda x: x if x > 0 else 0.01*x),
            'softplus': (lambda x: np.log(1 + e**x)),
            'gauss':    (lambda x: e**(-x**2))
             }

        # Iterates through each later except for last one to feed forward
        # node value = activation fn( (previous nodes * weights from previous nodes to current node) + bias )
        for i in range(self.n_layers - 1):
            next_nodes = np.dot(self.architecture[i].values, self.architecture[i].weights) + self.architecture[i].bias
            self.architecture[i+1].values = [
                activate[self.architecture[i+1].fn](node) for node in next_nodes
                ]

        return self.architecture[-1].values


    def backward_propagate(self, expected_value):
        """
        Performs backward propagation on every layer in the nextwork.
        Goind backward from last layer to the second layer (incl.) and
        calculates d, the partial derivative with respect to the previous layer.
        Then, finds the amount weights should be changed by via a product of
        the learning rate, previous d, and current node's value.
        Applies the changes to the weights.

        Args:
            expected_value: expected output value from the dataset

        Returns:
            instance_error: mean squared error from expected values and
            calculated output values.
        """
        # TODO: Add Momentum

        # Holds activation functions as lambda functions in a dictionary
        derivative = {
            'linear':   (lambda x: 1),
            'neglin':   (lambda x: -1),
            'inv':      (lambda x: -1/(x**2)),
            'squared':  (lambda x: 2*x),
            'cube':     (lambda x: 3*x**2),
            'sigmoid':  (lambda x: (1/(1+(e**(-x))) * (1 - 1/(1+(e**(-x)))))),
            'relu':     (lambda x: 1 if x > 0 else 0),
            'tanh':     (lambda x: 1 - np.tanh(x)**2),
            'leaky':    (lambda x: 1 if x > 0 else 0.01),
            'softplus': (lambda x: 1/(1 + e**(-x))),
            'gauss':    (lambda x: -2 * x * (e**(-x**2)))
        }

        # Iterates through layers backwards, excludes the input layer
        for i in range(self.n_layers-1, 0, -1):
            # Calculates d for the output layer and square error for the data instance
            # d = (expected value - calculated value) * derivative(predicted value)
            # square error = (expected value - calculated value)^2
            if i == self.n_layers-1:
                square_err = []
                for j in range(len(self.architecture[-1].values)):
                    predicted_output = self.architecture[-1].values[j]
                    self.architecture[-1].d[j] = (
                        (expected_value[j] - predicted_output)
                        * derivative[self.architecture[-1].fn](predicted_output) )
                    square_err.append((expected_value[j] - predicted_output)**2)
                instance_error = np.sum(square_err) / len(expected_value)  #mean square error
            # Calculates d for all the hidden layer nodes
            # d = (node weight from current node to next node * d of next layer's node) for every node
            else:
                for j in range(self.architecture[i].n):
                    prev_ds = self.architecture[i+1].d
                    node_val = self.architecture[i].values[j]
                    node_wts = self.architecture[i].weights[j]
                    sum_prod = sum( [node_wts[k] * prev_ds[k] for k in range(len(node_wts))] )
                    self.architecture[i].d[j] = sum_prod * derivative[self.architecture[i].fn](node_val)

        # For every layer and node in that layer, calculates the weight change and
        # applies it to the weight used in forward propagation
        for i in range(self.n_layers-1):

            for j in range(self.architecture[i].n):
                delta_wts = [
                    self.l_rate * self.architecture[i+1].d[k] * self.architecture[i].values[j]
                    for k in range(self.architecture[i+1].n)
                    ]
                self.architecture[i].weights[j] = [
                    self.architecture[i].weights[j][k]
                    + delta_wts[k]
                    for k in range(len(delta_wts))
                    ]

                self.architecture[i].bias += self.l_rate * sum(self.architecture[i+1].d)


        return instance_error


    def train(self, X, y, epochs, verbose=False, use_seed=False, decay=1, decay_rate=10, momentum=0, plot=False):
        """
        Trains the multilayer perceptron by propagating each data instance forward
        and backward through a number of epochs specified by the user.

        Args:
            X: training attribute data instances
            y: expected output for every data instance
            epochs: number of epochs to train the network
            verbose: flag for verbose output (default False)
            use_seed: flag to set a random() seed (default False)
            decay: learning rate decay (defaulted to no decay)
            decay_rate: number of epochs after which to decay the learning rate (defaulted to 10)
            plot: flag to show the learning curve plot (learning rate vs. number of epochs) (default to False)
        """
        epoch_errors = []  #initializes the error at each epoch
        if verbose == True: print('\nTRAINING RESULTS\n' + '%6s %10s %12s' %('EPOCH', 'ERROR', 'PERFORMANCE') + '\n' + '-'*30)

        # Iterates through epochs and feeds a data instance in X into forward
        # and backward propagation in the network.
        # Also keeps track of time from epoch start to end for performance analysis.
        for e in range(epochs):
            start = time.time()
            epoch_error = []

            # Learning rate decay if specified by user
            if (e % decay_rate == 0) and (e != 0):
                self.l_rate *= decay

            # Iterates through all data instances in the set
            for i in range(len(X)):
                self.architecture[0].values = X[i]          #sets the input values of the input layer to instance values
                self.forward_propagate()                    #performs forward propagation on the instance
                if type(y[i]) is not list: y[i] = [y[i]]    #standardizes output as a list, if output is an int or float
                inst_err = self.backward_propagate(y[i])    #performs backpropagation and returns the instance error
                epoch_error.append(inst_err)
            epoch_errors.append(np.mean(epoch_error))

            end = time.time()

            if verbose == True: print('%6d %10.5f %12.3f' %(e, np.mean(epoch_error), end-start))

        if verbose == True: print('-'*30)

        # If plot flag is on, returns a plot of learning rate vs. epoch number
        if plot == True:
            plt.plot([i for i in range(epochs)], epoch_errors)
            plt.title('Learning Curve')
            plt.xlabel('Epoch Number')
            plt.ylabel('Mean Absolute Error')
            plt.show()

        return epoch_errors[-1] #returns last training epoch error

    def predict(self, X, verbose=False, save_file=None):
        """
        Performs testing of the data instances by feeding them into the network
        and performing forward propagation with previously attained weights and bias.

        Args:
            X: testing attribute data instances
            verbose: flag for verbose output (default False)
            save_file: acts as flag; if specified by user as a string,
                saves predicted values into that csv file (default None)

        Returns:
            predicted_values: list of values predicted for every input instance
        """

        predicted_values = []  #initialize a list of predicted values
        if verbose == True: print('TESTING RESULTS')

        # Iterates through testing data instances and forward propagates the instances
        for i in range(len(X)):
            self.architecture[0].values = X[i]          #sets input layer values
            predicted_val = self.forward_propagate()    #forward propagates and returns the prediction
            predicted_values.append(predicted_val)      #adds prediction to list
            if verbose == True: print('INPUT:', X[i], '\tOUTPUT:', predicted_val)

        # If flag is on, saves the predictions into a csv file with specified file name
        if save_file != None:
            with open(save_file + '.csv', 'w', newline='') as results:
                w = csv.writer(results, delimiter=',')
                for i in range(len(predicted_values)):
                    w.writerow(predicted_values[i])

        return predicted_values



    def save(self, name='network'):
        """
        Saves network parameters, including number of layers, number of nodes per
        layer, activation function, weights, and bias into a text file,
        encoded using pickle library.

        Args:
            name: name of the file to which the network parameters are saved
                    (defaulted to 'network.txt')
        """

        with open(name + '.txt', 'wb') as f:
            pickle.dump(self.n_layers, f)
            # Iterates through every layer and saves
            # n = number of nodes, fn = activation function, weights, bias
            for i in range(self.n_layers):
                pickle.dump(self.architecture[i].n, f)
                pickle.dump(self.architecture[i].fn, f)
                pickle.dump(self.architecture[i].weights, f)
                pickle.dump(self.architecture[i].bias, f)
        print('> Network achitecture successfully saved to ' + name + '.txt.')


    def load(self, name):
        """
        Loads network parameters, including number of layers, number of nodes per
        layer, activation function, weights, and bias into a text file,
        encoded using pickle library.

        Args:
            name: name of the file to which the network parameters were saved
        """

        with open(name + '.txt', 'rb') as f:
            # Loads first python object as number of layers
            self.n_layers = pickle.load(f)
            # Iterates through pickled objects in file and retrieves
            # number of nodes, activation function, weights, and bias
            for i in range(self.n_layers):
                n = pickle.load(f)                  #get number of neurons in the layer
                self.architecture.append(Layer(n))  #initialize layer
                self.architecture[i].fn = pickle.load(f)
                self.architecture[i].weights = pickle.load(f)
                self.architecture[i].bias = pickle.load(f)
        print('> Network achitecture successfully loaded from ' + name + '.txt.')


    def show(self):
        """
        Prints network inforamtion on screen, including number of nodes in
        each layer, activation function, bias, node values, and weights.
        Does not print weights for the output layer.
        """

        for layer in self.architecture:
            print('\nLAYER','-' * 94)
            print('Number of Nodes:', layer.n)
            print('Activation Function:', layer.fn)
            print('Bias:', layer.bias)
            print('\nNODES:')
            for i in range(layer.n):
                if len(layer.weights) != 0:
                    print('Node Value:', layer.values[i], '\tNode Weights:', layer.weights[i])
                else:
                    print('Node Value:', layer.values[i])
            print('-' * 100)


    def visualize(self):
        """
        Visualizes network nodes and weights and saves the visualization into a
        png file 'graph.png'.
        """

        im = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(im)

        all_nodes = []
        layers = [((i+1) * 800)/(self.n_layers + 1) for i in range(self.n_layers)]

        # Circles
        for i in range(len(layers)):
            layer_nodes = []
            for j in range(self.architecture[i].n):
                x = layers[i]
                y = ((j+1)*600)/(self.architecture[i].n + 1)
                layer_nodes.append((x, y))
                draw.ellipse((x-20, y-20, x+20, y+20), fill='white', outline='black')
            all_nodes.append(layer_nodes)


        # Lines
        for i in range(len(all_nodes) - 1):  #except for last layer
            for c, node in enumerate(all_nodes[i]):
                for next_c, next_node in enumerate(all_nodes[i+1]):
                    draw.line((node, next_node), fill='black')

        del draw
        im.show()
        im.save('graph.png')
