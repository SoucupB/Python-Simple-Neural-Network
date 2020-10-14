import ctypes
import math
import time
from NeuralNetwork import NeuralNetwork, SIGMOID, TANH, RELU
SIGMOID = 0
TANH = 1
RELU = 2

class QAgent():
    def buffer_to_list(self, buffer, size):
        return_buffer = []
        for index in range(size):
            return_buffer.append(self.fun.elementFromBuffer(buffer, ctypes.c_int(index)))
        return return_buffer

    def convert_to_float_pointer(self, buffer):
        buffer_pointer = ctypes.c_float * len(buffer)
        buffer_arr = buffer_pointer(*buffer)
        return buffer_arr

    def convert_to_int32_pointer(self, buffer):
        buffer_pointer = ctypes.c_int32 * len(buffer)
        buffer_arr = buffer_pointer(*buffer)
        return buffer_arr

    def convert_float_array_pointer(self, buffer):
        buffer_pointer = ctypes.POINTER(ctypes.c_float) * len(buffer)
        buffer_arr = buffer_pointer(*buffer)
        return buffer_arr

    def __init__(self, neuralNet, lr, discountFactor, numberOfActions):
        self.fun = ctypes.CDLL("Components/NeuralNetwork.so")
        self.lr = lr
        self.neuralNet = neuralNet
        self.discountFactor = discountFactor
        self.numberOfActions = numberOfActions
        self.qa = self.fun.qa_Init(self.neuralNet.neuralNet, ctypes.c_float(self.lr), ctypes.c_float(self.discountFactor), ctypes.c_int(self.numberOfActions))
        self.fun.qa_GetChoosenActionIndex.restype = ctypes.c_int32

    def getBestAction(self, state, prohibitedStates):
        return self.fun.qa_GetChoosenActionIndex(self.qa, self.convert_to_float_pointer(state), self.convert_to_int32_pointer(prohibitedStates), len(prohibitedStates))

    def getBestActionWithRandomChance(self, state, prohibitedStates, chance):
        return self.fun.qa_GetActionWithRandom(self.qa, self.convert_to_float_pointer(state),
                                                self.convert_to_int32_pointer(prohibitedStates), len(prohibitedStates), ctypes.c_float(chance))

    def agentDestroy(self):
        self.fun.qa_Destroy(self.qa)

    def trainTemporalDifference(self, buffer, actions_taken, reward):
        # matrix_buffer = []
        # for index in range(len(buffer)):
        #     matrix_buffer.append(self.convert_to_float_pointer(buffer[index]))
        # self.fun.qa_TrainTemporalDifference(self.qa, self.convert_float_array_pointer(matrix_buffer),
        #                                     self.convert_to_int32_pointer(actions_taken), ctypes.c_float(reward), len(actions_taken))
        self.fun.qa_TrainTemporalDifferenceReplay(self.qa, ctypes.c_float(reward))

    def showReplay(self):
        self.fun.qa_ShowExperienceReplay(self.qa)

    def trainDeepQValue(self, buffer, actions_taken, reward):
        matrix_buffer = []
        for index in range(len(buffer)):
            matrix_buffer.append(self.convert_to_float_pointer(buffer[index]))
        self.fun.qa_TrainDeepQNet(self.qa, self.convert_float_array_pointer(matrix_buffer),
                                            self.convert_to_int32_pointer(actions_taken), self.convert_to_float_pointer(reward), len(actions_taken))

#nn = NeuralNetwork([2, 4, 4, 2], 0.1, [RELU, TANH, SIGMOID])
# nn = NeuralNetwork([18, 18, 1], 0.1, [RELU, SIGMOID])

# agent = QAgent(nn, 0.2, 0.1, 9)

# buffer = [[0, 0, 0, 0, 1, 0, 0, 0, 0],
#           [0, 0, 1, -1, 1, 0, 0, 0, 0],
#           [-1, 1, 1, -1, 1, 0, 0, 0, 0],
#           [-1, 1, 1, -1, 1, 0, -1, 1, 0]]

# actions_taken = [2, 1, 4, 0]

# print(agent.getBestAction([0, 0, 0, 0, 1, 0, 0, 0, 0], [4]))

# agent.trainTemporalDifference(buffer, actions_taken, 1)