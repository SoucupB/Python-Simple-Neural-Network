import time
import random
import NeuralNetwork as nn
import QAgent as qa

class Snake():
    def __init__(self, width, height, pos_x, pos_y):
        self.map = []
        self.width = width
        self.height = height
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.headPosX = pos_x
        self.headPosY = pos_y
        self.direction = 0
        self.init_map()
        self.foodX = -1
        self.foodY = -1
        self.score = 0
        self.bestScore = self.score
        self.randomPositions = []
        self.generate_determistic_food(100)
        self.tail = 0
        self.foodIndex = 0
        self.frame = 0
        self.neural = nn.NeuralNetwork([self.width * self.height * 2, 34, 1], 0.11, [nn.RELU, nn.SIGMOID])
        self.agent = qa.QAgent(self.neural, 0.6, 0.99, 4)
        self.game = 0

    def init_map(self):
        self.map = []
        for i in range(self.height):
            row = [0] * self.width
            self.map.append(row)

    def on_frame_food(self):
        self.foodX = self.randomPositions[self.foodIndex][0]
        self.foodY = self.randomPositions[self.foodIndex][1]

    def generate_determistic_food(self, numberOfSteps):
        for i in range(numberOfSteps):
            self.randomPositions.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))

    def print_state(self):
        for i in range(self.height):
            for j in range(self.width):
                print(self.cell(i, j), end = ' ')
            print()
        print("Current score ", self.score, " Best Score: ", self.bestScore)
        print()

    def is_dead(self):
        if self.headPosX < 0 or self.headPosY < 0 or self.headPosX >= self.width or self.headPosY >= self.height:
            return 1
        return 0

    def restart(self):
       # self.agent.trainTemporalDifferenceExpReplay(self.frame / 100 + self.score / 10)
        self.foodIndex = 0
        self.headPosX = self.pos_x
        self.headPosY = self.pos_y
        if self.score > self.bestScore:
            self.bestScore = self.score
        self.score = 0
        self.frame = 0
        self.game += 1
        self.on_frame_food()
    def cell(self, y, x):
        self.map[y][x] = 0
        if self.foodX == x and self.foodY == y:
            self.map[y][x] = 2
        if self.headPosX == x and self.headPosY == y:
            self.map[y][x] = 1
        return self.map[y][x]

    def eatFood(self):
        if self.headPosX == self.foodX and self.headPosY == self.foodY:
            self.score += 1
            self.tail += 1
            self.foodIndex += 1
            return 1
        return 0
    def move_up(self):
        self.headPosY -= 1
        self.direction = 0
    def move_down(self):
        self.headPosY += 1
        self.direction = 1
    def move_left(self):
        self.headPosX -= 1
        self.direction = 2
    def move_right(self):
        self.headPosX += 1
        self.direction = 3
    def on_frame_move(self):
        if self.direction == 0:
            self.move_up()
        if self.direction == 1:
            self.move_down()
        if self.direction == 2:
            self.move_left()
        if self.direction == 3:
            self.move_right()
    def get_state(self):
        state = []
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] == 0:
                    state.append(0)
                    state.append(0)
                if self.map[i][j] == 1:
                    state.append(0)
                    state.append(1)
                if self.map[i][j] == 2:
                    state.append(1)
                    state.append(0)
        return state

    def action_move(self):
       # direction = random.randint(0, 4)
        direction = self.agent.getBestActionWithRandomChance(self.get_state(), [], 0.04)
        self.move(direction)
    def move(self, direction):
        if self.direction == 0:
            if direction == 2 or direction == 3:
                self.direction = direction

        if self.direction == 1:
            if direction == 2 or direction == 3:
                self.direction = direction

        if self.direction == 2:
            if direction == 1 or direction == 0:
                self.direction = direction

        if self.direction == 3:
            if direction == 1 or direction == 0:
                self.direction = direction

    def on_frame(self):
        self.action_move()
        self.on_frame_move()
        self.eatFood()
        self.on_frame_food()
        if self.is_dead():
            self.restart()
        self.frame += 1

    def on_game(self):
        while True:
            self.action_move()
            self.on_frame_move()
            self.eatFood()
            self.on_frame_food()
            if self.is_dead():
                return True
                self.restart()
            self.frame += 1


# snake = Snake(8, 8, 3, 3)
# while True:
#     snake.on_game()
#     #snake.print_state()
#     snake.agent.trainTemporalDifferenceExpReplay(snake.frame / 100 + snake.score / 10)
#     if snake.game % 500 == 0:
#         print("Game ", snake.game, "Best score: ", snake.frame / 100 + snake.score / 10, "Max foods: ", snake.bestScore)
#     snake.restart()
#    # time.sleep(0.5)


arb = nn.NeuralNetwork([2, 5, 1], 0.1, [nn.RELU, nn.RELU])

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = [[1], [3], [5], [6]]

for i in range(6000):
    mike = [0, 1, 2, 3]
    random.shuffle(mike)
    for index in range(len(input)):
        #print(arb.feed_forward(input[index]))
        print(arb.sgd(input[mike[index]], output[mike[index]], nn.OPT_ADAGRAD))

for index in range(len(input)):
    print(arb.feed_forward(input[index]))