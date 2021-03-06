import NeuralNetwork as nn
from QAgent import QAgent
import math
import random
import matplotlib.pyplot as plt
#from Neat import *

def sigmoid(element):
    return 1.0 / (1.0 + math.exp( -element ))

def d_sigmoid(element):
    return sigmoid(element) * (1.0 - sigmoid(element))

def relu(element):
    if element <= 0:
        return 0.001
    return element
def d_relu(element):
    if element <= 0:
        return 0.001
    return 1

def tanh(element):
    return math.tanh(element)

def d_tanh(element):
    return 1 - tanh(element) * tanh(element)

def loss_functions(a, b): #error can be positive or negative, if this value is always positive, then the net will not train!
    c = 1
    if (a - b) < 0:
        c = -1
    return (a - b) * (a - b) * c

def tic_tac_state(game_map, player):
    if game_map[0] == game_map[4] and game_map[0] == game_map[8] and game_map[0] == player:
        return 1
    if game_map[2] == game_map[4] and game_map[4] == game_map[6] and game_map[2] == player:
        return 1
    for index in range(3):
        checker = 1
        for j in range(3):
            if game_map[j + index * 3] != player:
                checker = 0
                break
        if checker == 1:
            return 1

    for index in range(0, 3):
        checker = 1
        for j in range(3):
            if game_map[j * 3 + index] != player:
                checker = 0
                break
        if checker == 1:
            return 1
    return 0

def draw(game_map):
    for elem in game_map:
        if elem == 0:
            return 0
    return 1

def finish_state(game_map):
    return tic_tac_state(game_map, 1) | tic_tac_state(game_map, 2) | draw(game_map)

def random_player(game_map, player):
    response = []
    for index in range(len(game_map)):
        if game_map[index] == 0:
            response.append(index)
    return random.choice(response)

def bot_response(game_map):
    f = [1, 0, 0]
    s = [0, 1, 0]
    l = [0, 0, 1]
    bot_map = []
    for index in range(9):
        if game_map[index] == 1:
            bot_map += f
        if game_map[index] == 2:
            bot_map += s
        if game_map[index] == 0:
            bot_map += l
    return bot_map

def save_map(state, actions, total_states, move):
    c_state = [x for x in state]
    total_states.append(state)
    actions.append(move)

def actor_prediction(bot, game_map):
    bot_map = bot_response(game_map)
    return bot.feed_forward(bot_map)

def select_actor_action(game_map, distribution):
    max_value = -1e9
    action_index = 0
    for index in range(len(distribution)):
        if max_value < distribution[index] and game_map[index] == 0:
            max_value = distribution[index]
            action_index = index
    return action_index


def train_TD(agent, reward):
    agent.trainTemporalDifferenceExpReplay(reward, nn.OPT_SGD)

def train_QValues(total_states, actions, agent, reward):
    rew = [0] * len(total_states)
    rew[len(total_states) - 1] = reward
    agent.trainDeepQValue(total_states, actions, rew)

def train(agent, reward):
    train_TD(agent, reward)
   # train_QValues(total_states, actions, agent, reward)

def show_game(game_map):
    for i in range(3):
        for j in range(3):
            print(game_map[i * 3 + j], end = ' ')
        print()
    print(finish_state(game_map))

def game(agent, agent_x, display):
    game_map = [0] * 9
    f = 0
    s = 0
    d = 0
    while(finish_state(game_map) == 0):
        first_move = random_player(game_map, 1)
        #first_move = get_best_action(game_map, agent_x, 1, 1, 0.07)
        if game_map[first_move] != 0:
            return 2
        game_map[first_move] = 1
        if display == 1:
            show_game(game_map)
        if tic_tac_state(game_map, 1) == 1:
            return 1
        if draw(game_map) == 1:
            return 0
        second_move = get_best_action(game_map, agent, 2, 1, 0.07)
        #second_move = random_player(game_map, 2)
        if game_map[second_move] != 0:
            return 1
        game_map[second_move] = 2
        if display == 1:
            show_game(game_map)
        if tic_tac_state(game_map, 2) == 1:
            return 2
        if draw(game_map) == 1:
            return 0
    return -1

def get_best_action(game_map, bot, player, chanceflag, chance):
    prohibited_actions = []
    for index in range(len(game_map)):
        if game_map[index] != 0:
            prohibited_actions.append(index)
    if chanceflag == 1:
        return bot.getBestActionWithRandomChance(bot_response(game_map), prohibited_actions, chance)
    return bot.getBestAction(bot_response(game_map), prohibited_actions)

def batch_game(display, agent, agent_x, t_batches, index):
    batches = t_batches
    ind = 0
    f = 0
    s = 0
    d = 0
    while ind < batches:
        result = game(agent, agent_x, 0)
      #  agent.showReplay()
      #  exit()
        if result == 1:
            train(agent, 0)
            train(agent_x, 1)
            f += 1
        if result == 2:
            train(agent, 1)
            train(agent_x, 0)
            s += 1
        if result == 0:
            train(agent, 0.5)
            train(agent_x, 0.5)
            d += 1
        ind += 1
    if f == 0:
        print(index, t_batches)
    if f != 0:
        print(index, ((d / 2 + s) / f))
    return (s, f, d)

def plot(games, wins, draws):
    plt.xlabel('batches!')
    plt.ylabel('wins + draws of the "O" agent (per 1000 matches!)')
    plt.title("TicTacToe Deep Q Learning Agent vs random agent")
    plt.plot(games, wins)
    plt.savefig("Plots/TicTacToe_wins.png")
    return 0
def instance(order):
    total_batches = 700
    nets_number = 1
    number_of_games_per_batch = 1000
    critic_net = nn.NeuralNetwork([27, 51, 1], 0.13, [nn.RELU, nn.SIGMOID])
    critic_net_x = nn.NeuralNetwork([27, 35, 35, 1], 0.11, [nn.RELU, nn.RELU, nn.SIGMOID])
    agent = QAgent(critic_net, 0.4, 0.99, 9)
    agent_x = QAgent(critic_net_x, 0.4, 0.99, 9)
    games_number = []
    wins = []
    draws = []
    if order == 0:
        for index in range(total_batches):
            fitness = batch_game(0, agent, agent_x, number_of_games_per_batch, index)
            games_number.append((index + 1))
            wins.append(fitness[0] + fitness[2])
            draws.append(fitness[2])
        plot(games_number, wins, draws)
        critic_net.save_weights("netValues.rt")
    else:
        critic_net.load_weights("netValues.rt")
        game(agent, agent_x, 1)
instance(0)

#gcc -fPIC -shared NeuralNetwork.c hashmap.c Functions.c Neuron.c QAgent.c -Wall -o NeuralNetwork.so -O9
#gcc NeuralNetwork.c hashmap.c Functions.c Neuron.c MainXOR.c QAgent.c -o program -O9 -lm
#gcc NeuralNetwork.c hashmap.c Functions.c Neuron.c MainQTEST.c QAgent.c -o program -Wall -O9 -lm

