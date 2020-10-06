
import random
from Neuron import Neuron
class NeuralNetwork():
    def random_value(self):
        return random.uniform(-0.1, 0.1)

    def __init__(self, params, activation_function, d_activation_function):
        self.input_Neurons = []
        self.hidden = []
        self.lr = 0.3
        self.biases = []
        self.tie_values = {}
        self.Neuron_map = {}
        for index in range(params[0]):
            self.input_Neurons.append(Neuron(index, self.Neuron_map, self.tie_values, 0))
            self.input_Neurons[index].activation_function = activation_function
            self.input_Neurons[index].d_activation_function = d_activation_function
        offset = 0
        for index in range(1, len(params)):
            offset += params[index - 1]
            current_hiddens = []
            for current_layer in range(params[index]):
                current_hiddens.append(Neuron(current_layer + offset, self.Neuron_map, self.tie_values, 0))
                current_hiddens[current_layer].activation_function = activation_function
                current_hiddens[current_layer].d_activation_function = d_activation_function
            self.hidden.append(current_hiddens)
        offset += params[len(params) - 1]
        for i in range(len(self.input_Neurons)):
            for j in range(len(self.hidden[0])):
                self.input_Neurons[i].tie(self.hidden[0][j].ID, self.random_value())
        for i in range(len(self.hidden) - 1):
            for j in range(len(self.hidden[i])):
                for k in range(len(self.hidden[i + 1])):
                    self.hidden[i][j].tie(self.hidden[i + 1][k].ID, self.random_value())
        for index in range(len(self.hidden)):
            self.biases.append(Neuron(offset + index + 1, self.Neuron_map, self.tie_values, 1))
        for i in range(len(self.hidden)):
            for j in range(len(self.hidden[i])):
                self.biases[i].tie(self.hidden[i][j].ID, self.random_value())
    def clear_Neurons(self):
        for i in range(len(self.hidden)):
            for j in range(len(self.hidden[i])):
                self.hidden[i][j].value = 0
                self.hidden[i][j].error = 0
    def feed_forward(self, values):
        response = []
        self.clear_Neurons()
        for i in range(len(self.input_Neurons)):
            self.input_Neurons[i].value = values[i]
            self.input_Neurons[i].feed_forward()
        for i in range(len(self.hidden)):
            self.biases[i].feed_forward()
            for j in range(len(self.hidden[i])):
                self.hidden[i][j].value = self.hidden[i][j].activation_function_applyer()
                self.hidden[i][j].feed_forward()
        for i in range(len(self.hidden[len(self.hidden) - 1])):
            response.append(self.hidden[len(self.hidden) - 1][i].value)
        return response

    def sgd(self, inputs, outputs, loss_function):
        nn_response = self.feed_forward(inputs)
        total_error = 0
        for index in range(len(outputs)):
            loss_f = loss_function(outputs[index], nn_response[index])
            total_error += abs(loss_f)
            self.hidden[len(self.hidden) - 1][index].error = loss_f
        for index in range(len(self.hidden) - 1, -1, -1):
            for i in range(len(self.hidden[index])):
                self.hidden[index][i].calculate_proportional_error_of_parents()
        for index in range(len(self.hidden) - 1, -1, -1):
            for i in range(len(self.hidden[index])):
                self.hidden[index][i].optimize(self.lr)
        return total_error

    def show_weights(self):
        for i in range(len(self.input_Neurons)):
            for j in range(len(self.input_Neurons[i].childs)):
                print(self.tie_values[(self.input_Neurons[i].ID, self.input_Neurons[i].childs[j])], end = ' ')
            print()
        print()
        for i in range(len(self.hidden)):
            for Neuron in self.hidden[i]:
                for child in Neuron.childs:
                    print(self.tie_values[(Neuron.ID, child)], end = ' ')
                print()
            print()
