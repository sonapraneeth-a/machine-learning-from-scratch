# Import statements
import numpy as np
np.set_printoptions(precision=3, linewidth=100)
import time, math, pickle
from library.models.classification.neuralnet.layers import InputLayer, HiddenLayer, SoftMaxLayer
from library.metrics.classification import accuracy_score
from library.utils.file_utils import check_file_exists


class NeuralNetwork:
    """
    brief: Class for the implementation of Neural network (Multi layer Perceptron)
    """

    def __init__(self, num_classes, verbose=False, random_seed=1, learning_rate=0.01,
                 log=False, one_hot=True, inspect=False, batch='auto', tolerance=0.001,
                 momentum=0.9, max_iterations=1000, disp_step=10, debug=False, reg_const=0.01,
                 config = \
                 {'Layer1': {'hidden_units': 10,
                             'activation': 'relu',
                             'weights': 'random_normal',
                             'bias': 'random_normal'},
                  'OutputLayer': {'activation': 'relu',
                                  'weights': 'random_normal',
                                  'bias': 'random_normal'}
                 },
                 save_model=True, model_filename='./weights.txt'):
        self.verbose = verbose
        self.debug = debug
        self.log = log
        self.inspect = inspect # Do you want to inspect inner layers?
        np.random.seed(random_seed)
        self.learn_rate = learning_rate # Learning rate for gradient descent
        self.one_hot = one_hot # Do you want one hot output in neural network?
        self.momentum = momentum # Momentum for stochastic gradient descent
        self.max_num_iterations = max_iterations # Maximum no. of iterations to train the network
        self.display_step = disp_step
        self.config = config # Neural network configuration
        self.input_layer = None # Input Layer
        self.hidden_layers = [] # Hidden layers list
        self.output_layer = None # Output layer
        self.predict_layer = None # Final prediction layer
        self.num_classes = num_classes # Number of output classes
        self.num_layers = len(self.config.keys()) # No. of layers including outer layer
        if batch == 'auto': # Batch size
            self.batch_size = 128
        elif batch == 'None' or batch == '':
            self.batch_size = None
        else:
            self.batch_size = int(batch)
        self.train_loss = []
        self.train_acc = []
        self.validate_loss = []
        self.validate_acc = []
        self.tolerance = tolerance
        self.reg_const = reg_const
        self.save_model = save_model
        self.model_file = model_filename

    def setup(self, input_nodes, output_nodes):
        """
        Setup up the neural network with the given configuration in self.config
        :param input_nodes: Number of features in the input data 
        :param output_nodes: Number of class labels in one hot format
        :return: True
        """
        print('Making Input layer')
        # Initialize input layer for feeding the data
        self.input_layer = InputLayer(output_neurons=input_nodes, layer_name='Input Layer')
        # Make all the hidden layers with the mentioned configuration
        for layer_no in range(1, self.num_layers):
            curr_layer_key = 'Layer' + str(layer_no)
            print('Making Hidden layer : %2d' % layer_no)
            # Get the number of hidden units in previous layer
            if layer_no == 1:
                input_units = input_nodes
            else:
                input_units = self.config[prev_layer_key]['hidden_units']
            # Get the number of hidden units in current layer
            output_units = self.config[curr_layer_key]['hidden_units']
            # Determine the previous layer
            if layer_no == 1:
                prev_layer = self.input_layer
            else:
                prev_layer = self.hidden_layers[-1]
            # Make the hidden layer with necessary information acquired above
            self.hidden_layers.append(HiddenLayer(prev_layer=prev_layer,
                                                  input_neurons=input_units,
                                                  output_neurons=output_units,
                                                  activation_type=self.config[curr_layer_key]['activation'],
                                                  layer_name='Hidden layer '+str(layer_no),
                                                  learning_rate=self.learn_rate,
                                                  debug=self.debug, momentum=self.momentum,
                                                  reg_const=self.reg_const))
            prev_layer_key = 'Layer' + str(layer_no)
        input_units = self.config[prev_layer_key]['hidden_units']
        # Make the output layer
        print('Making Output layer')
        if self.one_hot is True:
            output_neurons = self.num_classes
        else:
            output_neurons = 1
        self.output_layer = HiddenLayer(prev_layer=self.hidden_layers[-1],
                                        input_neurons=input_units,
                                        output_neurons=output_neurons,
                                        activation_type=self.config['OutputLayer']['activation'],
                                        layer_name='Output layer',
                                        learning_rate=self.learn_rate,
                                        debug=self.debug, momentum=self.momentum,
                                        reg_const=self.reg_const)
        # This layer is just for predicting the class probabilities for one hot encoding labels
        print('Making Softmax prediction layer')
        self.predict_layer = SoftMaxLayer(prev_layer=self.output_layer,
                                          nodes=output_nodes,
                                          one_hot=self.one_hot,
                                          layer_name='Softmax predict layer')
        print('Number of hidden layers : ' + str(len(self.hidden_layers)))
        print('Total number of layers  : ' + str(self.num_layers))
        print('Neural network setup completed')
        print()
        return True

    def print_network_setup(self):
        """
        Print the network setup
        :return: True
        """
        print('Neural network setup')
        # Network parameters
        print('Learning rate           : %.4f' % self.learn_rate)
        print('Momentum                : %.4f' % self.momentum)
        print('Regularization constant : %.4f' % self.reg_const)
        print('Error Tolerance         : %.8f' % self.tolerance)
        print('Layers')
        # Input layer
        print('\tInput layer')
        print('\t\tLayer name            : %s' % self.input_layer.layer_name)
        print('\t\tHidden units (Output) : %2d' % self.input_layer.output_nodes)
        # Hidden layers
        for layer in range(self.num_layers-1):
            print('\tLayer %s'%(str(layer+1).zfill(2)))
            print('\t\tLayer name            : %s' % self.hidden_layers[layer].layer_name)
            print('\t\tHidden units (Input)  : %2d' % (self.hidden_layers[layer].layer_weight.shape[0]))
            print('\t\tHidden units (Output) : %2d' % (self.hidden_layers[layer].layer_weight.shape[1]))
            print('\t\tIntermediate weights  : %s' % (str(self.hidden_layers[layer].layer_weight.shape)))
            print('\t\tBias                  : %s' % (str(self.hidden_layers[layer].layer_bias.shape)))
            print('\t\tActivation            : %s' % self.hidden_layers[layer].activation_type)
        # Output layer
        print('\tOutput Layer')
        print('\t\tLayer name            : %s' % self.output_layer.layer_name)
        print('\t\tHidden units (Input)  : %2d' % (self.output_layer.layer_weight.shape[0]))
        print('\t\tHidden units (Output) : %2d' % (self.output_layer.layer_weight.shape[1]))
        print('\t\tIntermediate weights  : %s' % (str(self.output_layer.layer_weight.shape)))
        print('\t\tBias                  : %s' % (str(self.output_layer.layer_bias.shape)))
        print('\t\tActivation            : %s' % self.output_layer.activation_type)
        # Prediction layer -> Softmax probability predictions
        print('\tSoftmax layer')
        print('\t\tLayer name            : %s' % self.predict_layer.layer_name)
        print('\t\tHidden units (Input)  : %2d' % self.predict_layer.nodes)
        print()
        return True

    def __str__(self):
        """
        See print_network_setup()
        :return: 
        """
        self.print_network_setup()

    def save_model_to_file(self, filename=None):
        """
        Save the learned parameters to file
        :param filename: Filename in which the parameters have to be written
        :return: True if write succeeds
        """
        print('Saving neural network trained model to file')
        if filename is None:
            filename = self.model_file
        print('Saving model to file: %s' % filename)
        model_dict = dict()
        model_dict['Learning_rate'] = self.learn_rate
        model_dict['Momentum'] = self.momentum
        model_dict['Reg_const'] = self.reg_const
        model_dict['Tolerance'] = self.tolerance
        model_dict['Config'] = self.config
        model_dict['num_hidden_layers'] = len(self.hidden_layers)
        model_dict['num_classes'] = self.num_classes
        model_dict['Input_layer'] = dict()
        model_dict['Input_layer']['weights'] = None
        model_dict['Input_layer']['bias'] = None
        model_dict['Input_layer']['weights_init'] = 'standard_normal'
        model_dict['Input_layer']['bias_init'] = 'standard_normal'
        model_dict['Input_layer']['input_nodes'] = None
        model_dict['Input_layer']['output_nodes'] = self.input_layer.output_nodes
        model_dict['Input_layer']['layer_name'] = self.input_layer.layer_name
        model_dict['Output_layer'] = dict()
        model_dict['Output_layer']['weights'] = self.output_layer.layer_weight
        model_dict['Output_layer']['bias'] = self.output_layer.layer_bias
        model_dict['Output_layer']['weights_init'] = 'standard_normal'
        model_dict['Output_layer']['bias_init'] = 'standard_normal'
        model_dict['Output_layer']['input_nodes'] = self.output_layer.input_nodes
        model_dict['Output_layer']['output_nodes'] = self.output_layer.output_nodes
        model_dict['Output_layer']['layer_name'] = self.output_layer.layer_name
        model_dict['Output_layer']['activation_fn'] = self.output_layer.activation_type
        model_dict['Softmax_layer'] = dict()
        model_dict['Softmax_layer']['weights'] = self.predict_layer.layer_weight
        model_dict['Softmax_layer']['bias'] = self.predict_layer.layer_bias
        model_dict['Softmax_layer']['weights_init'] = 'standard_normal'
        model_dict['Softmax_layer']['bias_init'] = 'standard_normal'
        model_dict['Softmax_layer']['nodes'] = self.predict_layer.nodes
        model_dict['Softmax_layer']['layer_names'] = self.predict_layer.layer_name
        for index, layer in enumerate(self.hidden_layers):
            dict_key = 'Layer_' + str(index+1).zfill(2)
            model_dict[dict_key] = dict()
            model_dict[dict_key]['weights'] = layer.layer_weight
            model_dict[dict_key]['bias'] = layer.layer_bias
            model_dict[dict_key]['weights_init'] = None
            model_dict[dict_key]['bias_init'] = None
            model_dict[dict_key]['input_nodes'] = layer.input_nodes
            model_dict[dict_key]['output_nodes'] = layer.output_nodes
            model_dict[dict_key]['layer_name'] = layer.layer_name
            model_dict[dict_key]['activation_fn'] = layer.activation_type
        with open(filename, 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Model saved to file: %s' % filename)
        return True

    def load_model_from_file(self, filename=None):
        """
        Load model parameters from a file
        :param filename: Filename from which model parameters have to be loaded
        :return: True if loading succeeds and network is setup
        """
        if filename is None:
            filename = self.model_file
        if check_file_exists(filename):
            print('Restoring model from file: %s' % filename)
            with open(filename, 'rb') as handle:
                model_dict = pickle.load(handle)
            self.config = model_dict['Config']
            self.num_layers = len(self.config.keys())
            self.num_classes = model_dict['num_classes']
            self.learn_rate = model_dict['Learning_rate']
            self.momentum = model_dict['Momentum']
            self.reg_const = model_dict['Reg_const']
            self.tolerance = model_dict['Tolerance']
            self.setup(input_nodes=model_dict['Input_layer']['output_nodes'],
                       output_nodes=model_dict['Softmax_layer']['nodes'])
            for index, layer in enumerate(self.hidden_layers):
                dict_key = 'Layer_' + str(index+1).zfill(2)
                layer.layer_weight = model_dict[dict_key]['weights']
                layer.layer_bias = model_dict[dict_key]['bias']
            self.output_layer.layer_weight = model_dict['Output_layer']['weights']
            self.output_layer.layer_bias = model_dict['Output_layer']['bias']
            print('Model restored from file: %s' % filename)
        else:
            raise FileNotFoundError('Model File %s does not exist' % filename)
        return True

    def forwardprop(self, input_data, input_data_labels):
        """
        One pass of forward propagation of neural network
        :param input_data: Data on which training is done
        :param input_data_labels: Output labels of the corresponding data in one hot format
        :return: 
        """
        for layer in range(self.num_layers-1):
            if self.verbose is True:
                print('Forward propagating layer %d' % (layer+1))
            if layer == 0:
                self.input_layer.layer_value = input_data
                prev_layer = self.input_layer
            else:
                prev_layer = self.hidden_layers[layer-1]
            current_layer = self.hidden_layers[layer]
            current_layer.forward(input=prev_layer)
            if self.debug is True:
                current_layer.debug_layer(input=prev_layer)
            if self.inspect is True:
                print(type(current_layer))
                current_layer.inspect_layer()
        if self.verbose is True:
            print('Forward propagating to outer layer')
        current_layer = self.output_layer
        prev_layer = self.hidden_layers[-1]
        current_layer.forward(input=prev_layer)
        if self.debug is True:
            current_layer.debug_layer(input=prev_layer)
        if self.inspect is True:
            current_layer.inspect_layer()
        self.predict_layer.output = input_data_labels
        self.predict_layer.update()
        if self.debug is True:
            print('Weight of predict layer')
            print(self.predict_layer.layer_weight)
            print('Bias of predict layer')
            print(self.predict_layer.layer_bias)
            print('Labels at predict layer')
            print(self.predict_layer.output)
            print('Delta at predict layer')
            print(self.predict_layer.delta)
            print('Loss at predict layer')
            print(self.predict_layer.loss)
        if self.inspect is True:
            self.predict_layer.inspect_layer()
        return True

    def backwardprop(self):
        """
        Learn the parameters by flowing backwards.
        Implemetation of backward propagation using momentum stochastic gradient descent
        :return: 
        """
        if self.verbose is True:
            print('Total number of layers  : ' + str(self.num_layers))
            print('Number of hidden layers : ' + str(len(self.hidden_layers)))
        if self.verbose is True:
            print('Backward propagating output layer')
        current_layer = self.output_layer
        next_layer = self.predict_layer
        current_layer.backward(next_layer=next_layer)
        if self.debug is True:
            print('Weight of output layer')
            print(current_layer.layer_weight)
            print('Bias of output layer')
            print(current_layer.layer_bias)
            print('Delta at output layer')
            print(current_layer.delta)
        if self.inspect is True:
            current_layer.inspect_layer()
        if self.verbose is True:
            print('Backward propagating output layer completed')
        for layer in reversed(range(self.num_layers-1)):
            if self.verbose is True:
                print('Backward propagating layer %d' % layer)
            current_layer = self.hidden_layers[layer]
            if layer == self.num_layers-2:
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[layer+1]
            current_layer.backward(next_layer=next_layer)
            if self.debug is True:
                print('Weight of layer %d' % layer)
                print(current_layer.layer_weight)
                print('Bias of layer %d' % layer)
                print(current_layer.layer_bias)
                print('Delta at layer %d' % layer)
                print(current_layer.delta)
            if self.inspect is True:
                current_layer.inspect_layer()
            if self.verbose is True:
                print('Backward propagating layer %d completed' % layer)
        return True

    def train(self, train_data, train_one_hot_labels, train_class_labels=None,
              validate_data=None, validate_one_hot_labels=None, validate_class_labels=None):
        """
        Train the neural network
        :param train_data: Data on which training happens
        :param train_one_hot_labels: Labels for train data in one hot format
        :param train_class_labels: Labels for train data in number format
        :param validate_data: Data on which model is evaluated for performance
        :param validate_one_hot_labels: Labels for validate data in one hot format 
        :param validate_class_labels: Labels for validate data in number format 
        :return: 
        """
        print('Training the model for %d iterations' %self.max_num_iterations)
        start = time.time()
        epoch = 0
        converged = False
        prev_loss = 0
        while epoch != self.max_num_iterations+1 and not converged:
            iter_start = time.time()
            # Do Forward Propagation
            if self.log is True:
                print('>> Epoch: %4d' %(epoch+1) )
            start_batch_index = 0
            if self.batch_size is None:
                self.batch_size = train_data.shape[0]
            num_batches = int(train_data.shape[0] / self.batch_size)
            print('Training done with %d batches' % num_batches)
            for batch in range(num_batches):
                end_batch_index = start_batch_index + self.batch_size
                if end_batch_index < train_data.shape[0]:
                    train_batch_data = train_data[start_batch_index:end_batch_index, :]
                    train_batch_one_hot_labels = train_one_hot_labels[start_batch_index:end_batch_index, :]
                else:
                    train_batch_data = train_data[start_batch_index:, :]
                    train_batch_one_hot_labels = train_one_hot_labels[start_batch_index:, :]
                if self.verbose is True:
                    print('Training on batch %d with %d samples' % (batch, train_batch_data.shape[0]))
                if self.verbose is True:
                    print('Forward Propagation for neural network started')
                self.forwardprop(train_batch_data, train_batch_one_hot_labels)
                if self.verbose is True:
                    print('Forward Propagation for neural network completed')
                # Do Backward Propagation
                if self.verbose is True:
                    print('Backward Propagation for neural network started')
                self.backwardprop()
                if self.verbose is True:
                    print('Backward Propagation for neural network completed')
                start_batch_index += self.batch_size
            iter_end = time.time()
            if train_class_labels is not None:
                train_loss, train_acc = \
                    self.evaluate(train_data, train_one_hot_labels, train_class_labels)
                if self.log is True:
                    print('train loss: %3.4f | train acc: %3.4f | time: %.4f s' %
                          (self.predict_layer.loss, train_acc, iter_end-iter_start))
                self.train_loss.append(train_loss)
                self.train_acc.append(train_acc)
            else:
                if self.log is True:
                    print('train_loss: %3.4f' % self.predict_layer.loss)
            if validate_data is not None and validate_class_labels is not None and \
                            validate_one_hot_labels is not None:
                validate_loss, validate_acc = \
                    self.evaluate(validate_data, validate_one_hot_labels, validate_class_labels)
                if self.log is True:
                    print('val. loss : %3.4f | val. acc : %3.4f | time: %.4f s' %
                          (validate_loss, validate_acc, iter_end-iter_start))
                self.validate_loss.append(validate_loss)
                self.validate_acc.append(validate_acc)
            epoch += 1
        end = time.time()
        print('Time taken to train neural network is %.4f seconds' % (end - start))
        print()
        return True

    def predict(self, data):
        """
        Generate predictions for the data
        :param data: Data for which class labels have to predicted
        :return: Probabilties of classes, Class labels and their corresponding one hot labels
        """
        if self.verbose is True:
            print('Making Predictions')
        for layer in range(self.num_layers - 1):
            if self.verbose is True:
                print('Forward propagating layer %d' % (layer + 1))
            if layer == 0:
                output = self.hidden_layers[layer].value(input=data)
            else:
                output = self.hidden_layers[layer].value(input=output)
        if self.verbose is True:
            print('Forward propagating to outer layer')
        output = self.output_layer.value(input=output)
        one_hot_labels = self.predict_layer.predict_one_hot_labels(input=output)
        class_labels = self.predict_layer.predict_class_labels(input=output)
        probs = self.predict_layer.predict_prob(input=output)
        return probs, class_labels, one_hot_labels

    def evaluate(self, data, data_one_hot_labels, data_class_labels):
        """
        Evaluate the model on the given data
        :param data: Data on which evaluation is done
        :param data_one_hot_labels: One hot labels of the data
        :param data_class_labels: Class labels of the data
        :return: 
        """
        # Performing forward propagation
        for layer in range(self.num_layers - 1):
            if self.verbose is True:
                print('Forward propagating layer %d' % (layer + 1))
            if layer == 0:
                output = self.hidden_layers[layer].value(input=data)
            else:
                output = self.hidden_layers[layer].value(input=output)
        if self.verbose is True:
            print('Forward propagating to outer layer')
        output = self.output_layer.value(input=output)
        one_hot_labels = self.predict_layer.predict_one_hot_labels(input=output)
        class_labels = self.predict_layer.predict_class_labels(input=output)
        probs = self.predict_layer.predict_prob(input=output)
        loss = self.predict_layer.get_loss(input=output, output_labels=data_one_hot_labels)
        accuracy = accuracy_score(data_class_labels, class_labels)
        return loss, accuracy
