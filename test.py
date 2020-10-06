from NeuralNetwork import NeuralNetwork
import math

def loss_functions(a, b): #error can be positive or negative, if this value is always positive, then the net will not train!
    c = 1
    if (a - b) < 0:
        c = -1
    return (a - b) * (a - b) * c

def sigmoid(element):
    return 1.0 / (1.0 + math.exp( -element ))

def d_sigmoid(element):
    return sigmoid(element) * (1.0 - sigmoid(element))

def tanh(element):
    return math.tanh(element)

def d_tanh(element):
    return 1 - tanh(element) * tanh(element)

def relu(element):
    if element <= 0:
        return 0.001
    return element
def d_relu(element):
    if element <= 0:
        return 0.001
    return 1

def sinc(element):
    return math.sin(element)
def d_sinc(element):
    return math.cos(element)

def gaussian(element):
    return math.exp(-(element * element))
def d_gaussian(element):
    return -2 * element * gaussian(element)


def main():
    print("Train XOR Problem!!")
    """
    You can uncomment lines for testing the Neural Network with different activation functions
    """
    args = NeuralNetwork([2, 2, 1], sigmoid, d_sigmoid)
   # args = NeuralNetwork([2, 2, 1], tanh, d_tanh)
   # args = NeuralNetwork([2, 2, 1], relu, d_relu)
   # args = NeuralNetwork([2, 2, 1], sinc, d_sinc)
   # args = NeuralNetwork([2, 2, 1], gaussian, d_gaussian)

    args.show_weights()
    print(args.feed_forward([1, 1]))
    for index in range(10000): #modify this number of batches in order to train more or less
        sr = [[1, 0], [0, 1], [1, 1], [0, 0]]
        pl = [[1], [1], [0], [0]]
        for i in range(len(sr)):
            print(args.sgd(sr[i], pl[i], loss_functions))
    print("For [1, 1], the net says is", args.feed_forward([1, 1]))
    print("For [1, 0], the net says is", args.feed_forward([1, 0]))
    print("For [0, 1], the net says is", args.feed_forward([0, 1]))
    print("For [0, 0], the net says is", args.feed_forward([0, 0]))

if __name__ == "__main__":
    main()