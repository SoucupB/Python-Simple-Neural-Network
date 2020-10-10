from CTest import NeuralNetwork, SIGMOID, RELU, TANH
import math
import random
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
    f = [1, 0]
    s = [0, 1]
    l = [0, 0]
    bot_map = []
    for index in range(9):
        if game_map[index] == 1:
            bot_map += f
        if game_map[index] == 2:
            bot_map += s
        if game_map[index] == 0:
            bot_map += l
    return bot_map

def save_map(state, total_states, move):
    c_state = [x for x in state]
    total_states.append([bot_response(c_state), move])

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

def train(total_states, actor_bot, critic_net, response, total_actions):
    goul = [0] * 9
    goul[total_states[len(total_states) - 1][1]] = 1
    critic_net.sgd(total_states[len(total_states) - 1][0] + goul, [response])
    value_vec = [0] * len(total_states)
    lr = 0.1
    value_vec[len(total_states) - 1] = response
    for elements in range(len(total_states) - 2, -1, -1):
        goul = [0] * 9
        goul[total_states[elements][1]] = 1
        feed_resp = critic_net.feed_forward(total_states[elements][0] + goul)
        value = (value_vec[elements + 1] - feed_resp[0]) * lr
        critic_net.sgd(total_states[elements][0] + goul, [value])
        value_vec[elements] = value
    for elements in range(len(total_states) - 1, -1, -1):
        goul = [0] * 9
        max_value = -1e9
        best_index = 0
        for index in range(total_actions):
            goul[index] = 1
            current_value = critic_net.feed_forward(total_states[elements][0] + goul)
            if current_value[0] > max_value:
                max_value = current_value[0]
                best_index = index
            goul[index] = 0
        goul[best_index] = 1
       # print(total_states[elements][0], goul)
        actor_bot.sgd(total_states[elements][0], goul)
   # print()
   # exit()



def show_game(game_map):
    for i in range(3):
        for j in range(3):
            print(game_map[i * 3 + j], end = ' ')
        print()
    print(finish_state(game_map))

def game(states, net, display):
    game_map = [0] * 9
    f = 0
    s = 0
    d = 0
    while(finish_state(game_map) == 0):
        first_move = random_player(game_map, 1)
        bot_map = bot_response(game_map)
        print(bot_map, net.feed_forward(bot_map))
        exit()
        if game_map[first_move] != 0:
            return 2
        game_map[first_move] = 1
        if display == 1:
            show_game(game_map)
        if tic_tac_state(game_map, 1) == 1:
            return 1
        if draw(game_map) == 1:
            return 0
        second_move = get_best_action(game_map, net, 2)
        save_map(game_map, states, second_move)
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

def get_best_action(game_map, bot, player):
    return select_actor_action(game_map, actor_prediction(bot, game_map))
    # maximum = -1e9
    # bot_map = bot_response(game_map)
    # value = bot.feed_forward(bot_map)
    # best_index = 0
    # for guess in range(len(value)):
    #     if game_map[guess] > maximum and game_map[guess] == 0:
    #         maximum = game_map[guess]
    #         best_index = guess
    # return best_index

def batch_game(display, actor_net, t_batches, critic_net):
    batches = t_batches
    ind = 0
    f = 0
    s = 0
    d = 0
    while ind < batches:
        states = []
        result = game(states, actor_net, 0)
        if result == 1:
            train(states, actor_net, critic_net, 0, 9)
            f += 1
        if result == 2:
            train(states, actor_net, critic_net, 1, 9)
            s += 1
        if result == 0:
            train(states, actor_net, critic_net, 0.5, 9)
            d += 1
        ind += 1
    if f == 0:
        print(10)
    print(((d / 2 + s) / f))
    if f == 0:
        return 10
    return ((d / 2 + s) / f)


total_batches = 15
nets_number = 1
#npt = Neat(nets_number, [9, 9], 0.07, 0.01)
actor_net = NeuralNetwork([18, 28, 28, 9], 0.09, [RELU, RELU, SIGMOID])# sigmoid, d_sigmoid)
#actor_net.load_bot("actor.tr")
critic_net = NeuralNetwork([27, 27, 27, 1], 0.08, [RELU, RELU, SIGMOID])#, sigmoid, d_sigmoid)
#actor_net.lr = 0.1
#critic_net.lr = 0.13

# for index in range(total_batches):
#     for crt in range(nets_number):
#         fitness = batch_game(0, actor_net, 1000, critic_net)
      #  npt.add_fitness(crt, fitness)
    #npt.next_population()
#actor_net.load_bot("actor.tr")
#actor_net = NeuralNetwork([18, 23, 9], sigmoid, d_sigmoid)
# # actor_net.save_bot("actor.tr")
actor_net.load_weights()
game([], actor_net, 1)

#actor_net.save_weights()

#gcc -fPIC -shared NeuralNetwork.c hashmap.c Functions.c Neuron.c QAgent.c -o NeuralNetwork.so -O3
#gcc NeuralNetwork.c hashmap.c Functions.c Neuron.c MainXOR.c QAgent.c -o program -O9 -lm
#gcc NeuralNetwork.c hashmap.c Functions.c Neuron.c MainQTEST.c QAgent.c -o program -Wall -O9 -lm

