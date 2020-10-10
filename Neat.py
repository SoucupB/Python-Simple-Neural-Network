from NeuralNetwork import *
import math
import random
import copy

class Neat():
    def sigmoid(self, element):
        return 1.0 / (1.0 + math.exp( -element ))

    def d_sigmoid(self, element):
        return self.sigmoid(element) * (1.0 - self.sigmoid(element))

    def __init__(self, number_of_nets, inps_outs, mutation_rate, mutation_value):
        self.number_of_nets = number_of_nets
        self.nets = []
        self.mutation_rate = mutation_rate
        self.mutation_value = mutation_value
        self.max_overal_net = -1e9
        self.fitness = [0] * number_of_nets
        for index in range(number_of_nets):
            self.nets.append(NeuralNetwork([inps_outs[0], 20, inps_outs[1]], self.sigmoid, self.d_sigmoid))

    def get_response_from(self, net_index, inputs):
        return self.nets[net_index].feed_forward(inputs)

    def add_fitness(self, index, value):
        self.fitness[index] = value

    def pick_net(self, fitness):
        index = 0
        total_number = sum(fitness)
        ftns = copy.deepcopy(fitness)
        random_nr = random.uniform(0, 1)
        for elemns in range(len(ftns)):
            ftns[elemns] /= total_number
        while random_nr > 0:
            random_nr -= ftns[index]
            index += 1

        return index - 1

    def next_population(self):
        # max_index = -1e9
        # c_index = 0
        tex = -1e9
        lindex = 0
        next_nets = []
        # for index in range(len(self.fitness)):
        #     if self.fitness[index] > max_index:
        #         max_index = self.fitness[index]
        #         c_index = index
        for index in range(self.number_of_nets):
            max_index = self.pick_net(self.fitness)
            if max_index >= len(self.fitness):
                print(max_index)
                exit()
            if tex < self.fitness[max_index]:
                tex = self.fitness[max_index]
                lindex = max_index
            next_net = copy.deepcopy(self.nets[max_index])
            next_net.mutate(self.mutation_rate, self.mutation_value)
            next_nets.append(next_net)
        if self.max_overal_net <= self.fitness[lindex]:
            self.max_overal_net = self.fitness[lindex]
            self.nets[lindex].save_bot("best_bot.tr")
        self.nets = next_nets
        print("Next population selected!")
        print(sum(self.fitness), "Max is ", self.fitness[max_index])
        self.fitness = [0] * self.number_of_nets