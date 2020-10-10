import ctypes
import math
import time
SIGMOID = 0
TANH = 1
RELU = 2

class NeuralNetwork():
    def __init__(self, input_array, lr, configuration):
        self.fun = ctypes.CDLL("NeuralNetwork.so")
        self.input_array = input_array
        self.lr = lr
        self.configuration = configuration
        ints_array = ctypes.c_int * len(input_array)
        configuration_array = ctypes.c_int * len(configuration)
        raw_array = ints_array(*self.input_array)
        configuration_array = configuration_array(*self.configuration)
        c_lr = ctypes.c_float(self.lr)
        self.neuralNet = self.fun.nn_InitMetaParameters(raw_array, len(self.input_array), c_lr, configuration_array)
        self.fun.elementFromBuffer.restype = ctypes.c_float
        self.fun.nn_Optimize.restype = ctypes.c_float
        self.fun.func_Uniform.restype = ctypes.c_float

    def show_weights(self):
        self.fun.nn_ShowWeights(self.neuralNet)

    def buffer_to_list(self, buffer, size):
        return_buffer = []
        for index in range(size):
            return_buffer.append(self.fun.elementFromBuffer(buffer, ctypes.c_int(index)))
        return return_buffer

    def feed_forward(self, inputs):
        c_inputs = ctypes.c_float * len(inputs)
        input_array = c_inputs(*inputs)
        response = self.fun.nn_FeedForward(self.neuralNet, input_array, len(input_array))
        arr = ctypes.c_float * 1
        list_of_results = self.buffer_to_list(response, self.input_array[len(self.input_array) - 1])
        return list_of_results

    def sgd(self, input, output):
        c_inputs = ctypes.c_float * len(input)
        input_array = c_inputs(*input)

        c_output = ctypes.c_float * len(output)
        output_array = c_output(*output)
        return self.fun.nn_Optimize(self.neuralNet, input_array, len(input), output_array, len(output))

    def destroy_nn(self):
        self.fun.nn_Destroy(self.neuralNet)

    def save_weights(self):
        self.fun.nn_WriteFile(self.neuralNet)

    def load_weights(self):
        self.fun.nn_LoadFile(self.neuralNet)

    def get_random(self, a, b):
        return self.fun.func_Uniform(ctypes.c_float(a), ctypes.c_float(b))

nn = NeuralNetwork([2, 4, 4, 2], 0.1, [RELU, TANH, SIGMOID])
# millis = int(round(time.time() * 1000))
# for index in range(18000): #modify this number of batches in order to train more or less
#     sr = [[1, 0], [0, 1], [1, 1], [0, 0]]
#     pl = [[1, 1], [1, 0], [0, 0], [0, 1]]
#     for i in range(len(sr)):
#         nn.sgd(sr[i], pl[i])

# print("Ended in! ", int(round(time.time() * 1000)) - millis)
# print("For [1, 1], the net says is", nn.feed_forward([1, 1]))
# print("For [1, 0], the net says is", nn.feed_forward([1, 0]))
# print("For [0, 1], the net says is", nn.feed_forward([0, 1]))
# print("For [0, 0], the net says is", nn.feed_forward([0, 0]))
# nn.save_weights()

# nn.destroy_nn()

print(nn.get_random(4, 9))