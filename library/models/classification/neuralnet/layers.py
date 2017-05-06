import numpy as np
np.set_printoptions(precision=3, linewidth=100)
from library.models.classification.neuralnet.params import weights, biases
from library.models.classification.neuralnet.activations import ReLU, Sigmoid, Tanh


def activate_layer(input_layer, activation_type='', verbose=False):
    """
    Activating the layer with the specified activation function
    :param input_layer: Layer to be activated
    :param activation_type: Activation function to be used
    :param verbose: If verbose is required
    :return: 
    """
    if activation_type == 'relu':
        activate = ReLU.value(input_layer)
    elif activation_type == 'sigmoid':
        activate = Sigmoid.value(input_layer)
    elif activation_type == 'tanh':
        activate = Tanh.value(input_layer)
    elif activation_type == '' or activation_type is None:
        activate = input_layer
    else:
        raise ValueError('Unknown Activation type')
    return activate


def activate_layer_deriv(input_layer, activation_type='', verbose=False):
    """
    Calculating the derivative of input layer wrt activation function for backward propagation
    :param input_layer: Layer to be activated
    :param activation_type: Activation function to be used 
    :param verbose: If verbose output is required
    :return: 
    """
    if activation_type == 'relu':
        activate = ReLU.derivative(input_layer)
    elif activation_type == 'sigmoid':
        activate = Sigmoid.derivative(input_layer)
    elif activation_type == 'tanh':
        activate = Tanh.derivative(input_layer)
    elif activation_type == '' or activation_type is None:
        activate = input_layer
    else:
        raise ValueError('Unknown Activation type')
    return activate


class HiddenLayer:
    """
    Class for HiddenLayer object
    """

    def __init__(self, input_neurons, output_neurons, prev_layer=None,
                 activation_type='', layer_name='Layer', learning_rate=0.1,
                 momentum=0.9, debug=False, inspect=False, reg_const=0.01,
                 verbose=False):
        self.verbose = verbose
        self.debug = debug
        self.inspect = inspect
        self.input_nodes = input_neurons
        self.output_nodes = output_neurons
        self.layer_weight = weights((input_neurons,output_neurons))
        self.layer_bias = biases((output_neurons,))
        self.prev_layer = prev_layer
        self.layer_value = None
        self.next_layer = None
        self.delta = None
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.last_gradient = None
        self.layer_name = layer_name
        self.reg_const= reg_const

    def forward(self, input=None):
        """
        Forward propagate the layer with given input
        :param input: Given input data
        :return: 
        """
        if input is None:
            input = self.prev_layer.layer_value
        self.layer_value = np.dot(input.layer_value, self.layer_weight) \
                           + self.layer_bias
        activated_layer = activate_layer(self.layer_value,
                                         activation_type=self.activation_type)
        self.layer_value = activated_layer
        if self.debug is True:
            self.debug_layer()
        return True

    def value(self, input):
        """
        Get the values of layer nodes
        :param input: Input to the layer
        :return: 
        """
        current_layer = np.dot(input, self.layer_weight) \
                             + self.layer_bias
        activated_layer = activate_layer(current_layer,
                                         activation_type=self.activation_type)
        return activated_layer

    def backward(self, next_layer=None):
        """
        Backward propagate the errors
        :param next_layer: 
        :return: 
        """
        if next_layer is None:
            next_layer = self.next_layer.layer_value
        derivative = activate_layer_deriv(self.layer_value, activation_type=self.activation_type)
        self.delta = np.dot(next_layer.delta, next_layer.layer_weight.T)
        self.delta = np.multiply(self.delta, derivative)
        # print(self.prev_layer.layer_value.shape[0])
        alpha = self.learning_rate / self.prev_layer.layer_value.shape[0]
        error_gradient = np.dot(self.prev_layer.layer_value.T, self.delta)
        if self.last_gradient is None:
            gradient = - (alpha * error_gradient)
        else:
            gradient = - (alpha * error_gradient) + (self.momentum * self.last_gradient)
        if self.reg_const > 0.0:
            gradient = gradient - (alpha * self.reg_const * self.layer_weight)
        self.last_gradient = gradient
        self.layer_weight = self.layer_weight + gradient
        self.layer_bias = self.layer_bias - (alpha * np.mean(self.delta, axis=0))
        return True

    def inspect_layer(self):
        """
        Inspect layer parameters
        :return: 
        """
        if self.inspect is True:
            print('Inspecting layer : %s' % self.layer_name)
            print('Layer weight: %s' % (str(self.layer_weight.shape)))
            print(self.layer_weight)
            print('Layer bias %s' % (str(self.layer_bias.shape)))
            print(self.layer_bias)
            print('Node values: %s' % str(self.layer_value.shape))
            print(self.layer_value)
        return True

    def debug_layer(self):
        """
        Debug layer parameters for proper weights and biases
        :return: 
        """
        if self.debug is True:
            print('Weight of layer: %s' % self.layer_name)
            print(self.layer_weight)
            print('Bias of layer: %s' % self.layer_name)
            print(self.layer_bias)
            if input is not None:
                print('Input to layer: %s' % self.layer_name)
                print(input.layer_value)
            print('Output of layer: %s' % self.layer_name)
            print(self.layer_value)

    def dropout(self):
        return True


class SoftMaxLayer:
    """
    Prediction layer to predict probabilties of classes
    """

    def __init__(self, prev_layer, nodes, output_labels=None, layer_name='Softmax layer',
                 one_hot=True, debug=False, inspect=False, verbose=False):
        self.verbose = verbose
        self.output = output_labels
        self.prev_layer = prev_layer
        self.layer_value = None
        self.delta = None
        self.layer_name = layer_name
        if one_hot is True:
            self.nodes = nodes
        else:
            self.nodes = 1
        self.layer_weight = np.identity(self.nodes)
        self.layer_bias = np.ones((self.nodes,))
        self.one_hot = one_hot
        self.loss = 0
        self.debug = debug
        self.inspect = inspect

    def inspect_layer(self):
        """
        Inspect layer parameters
        :return: 
        """
        print('Inspecting softmax layer : %s' %self.layer_name)
        print('Layer delta: %s' %str(self.delta.shape))
        print(self.delta)
        print('Expected output: %s' %str(self.output.shape))
        print(self.output)
        print('Loss: %.4f' % self.loss)
        return True

    def debug_layer(self):
        """
        Debug layer parameters for proper weights and biases
        :return: 
        """
        if self.debug is True:
            print('Weight of layer: %s' % self.layer_name)
            print(self.layer_weight)
            print('Bias of layer: %s' % self.layer_name)
            print(self.layer_bias)
            if input is not None:
                print('Input to layer: %s' % self.layer_name)
                print(input.layer_value)
            print('Output of layer: %s' % self.layer_name)
            print(self.layer_value)
        return True

    def softmax(self, x):
        if self.one_hot is True:
            e = np.exp(x - np.max(x))
            if e.ndim == 1:
                answer = e / np.sum(e, axis=0)
            else:
                answer = e / np.array([np.sum(e, axis=1)]).T
            self.layer_value = answer
        else:
            answer = x
        return answer

    def predict_prob(self, input=None):
        if input is None:
            input = self.prev_layer.layer_value
        return self.softmax(input)

    def predict_one_hot_labels(self, input=None):
        if input is None:
            input = self.prev_layer.layer_value
        probs = self.softmax(input)
        one_hot_labels = probs / np.max(probs)
        return one_hot_labels

    def predict_class_labels(self, input=None):
        if input is None:
            input = self.prev_layer.layer_value
        probs = self.softmax(input)
        class_labels = np.argmax(probs, axis=1)
        return class_labels

    def update(self, input=None):
        if input is None:
            input = self.prev_layer.layer_value
        probs = self.softmax(input)
        if self.one_hot is True:
            self.delta = -(self.output - probs)
        else:
            self.delta = np.dot(-(self.output - probs), self.layer_weight.T)
        # self.loss = 0.5*np.sum(np.power((self.output - probs), 2))
        self.loss = -np.mean(self.output * np.log(probs) + (1-self.output) * np.log(1-probs))

    def get_loss(self, input=None, output_labels=None):
        if input is None:
            input = self.prev_layer.layer_value
        if output_labels is None:
            output_labels = self.output
        probs = self.softmax(input)
        delta = -(output_labels - probs)
        # loss = 0.5*np.sum(np.power((output_labels - probs), 2))
        loss = -np.mean(output_labels*np.log(probs) + (1-output_labels)*np.log(1 - probs))
        return loss


class InputLayer:

    def __init__(self, output_neurons, input_neurons=None, data=None, layer_name='Input layer', verbose=False):
        self.verbose = verbose
        self.layer_value = data
        self.layer_name = layer_name
        self.input_nodes = None
        self.output_nodes = output_neurons

    def value(self):
        return self.layer_value
