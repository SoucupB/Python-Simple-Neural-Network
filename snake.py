# Simple Snake Game in Python 3 for Beginners
# By @TokyoEdTech

import turtle
import time
import random
import NeuralNetwork as nn
import QAgent as qa
import math

speed = 0.01
delay = speed

# Score
score = 0
high_score = 0
width = 600
height = 600
mex = 1000
totalFrames = mex

# Set up the screen
wn = turtle.Screen()
wn.title("Snake Game by @TokyoEdTech")
wn.bgcolor("green")
wn.setup(width=width, height=height)
wn.tracer(0) # Turns off the screen updates

# Snake head
head = turtle.Turtle()
head.speed(0)
head.shape("square")
head.color("black")
head.penup()
head.goto(0,0)
head.direction = "stop"

# Snake food
food = turtle.Turtle()
food.speed(0)
food.shape("circle")
food.color("red")
food.penup()
food.goto(0,100)

segments = []

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.shape("square")
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Score: 0  High Score: 0", align="center", font=("Courier", 24, "normal"))

# Functions
def go_up():
    if head.direction != "down":
        head.direction = "up"

def go_down():
    if head.direction != "up":
        head.direction = "down"

def go_left():
    if head.direction != "right":
        head.direction = "left"

def go_right():
    if head.direction != "left":
        head.direction = "right"

def convertCoords(y, x):
    deltaX = width // 20
    deltaY = height // 20
    return (deltaY - (y // 20 + 15), x // 20 + 15)

def transState(state):
    stateValues = []
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                stateValues.append(0)
                stateValues.append(0)
            if state[i][j] == 1:
                stateValues.append(0)
                stateValues.append(1)
            if state[i][j] == 2:
                stateValues.append(1)
                stateValues.append(0)
            if state[i][j] == 3:
                stateValues.append(1)
                stateValues.append(1)
    return stateValues

def state(width, height):
    deltaX = width // 20 + 1
    deltaY = height // 20 + 1
    matrix = []
    for i in range(deltaX):
        matrix.append([0 for y in range(deltaY)])
    y = head.ycor()
    x = head.xcor()
    food_y = food.ycor()
    food_x = food.xcor()
    newSelf = convertCoords(y, x)
    newFood = convertCoords(food_y, food_x)
    matrix[newSelf[0]][newSelf[1]] = 1
    matrix[newFood[0]][newFood[1]] = 2
    for elems in segments:
        xx = elems.xcor()
        yy = elems.ycor()
        resp = convertCoords(yy, xx)
        matrix[resp[0]][resp[1]] = 3
    # for i in range(deltaX):
    #     for j in range(deltaY):
    #         print(matrix[i][j], end = ' ')
    #     print()
    # print()
    return matrix

neural = nn.NeuralNetwork([31 * 31 * 2, 34, 1], 0.02, [nn.RELU, nn.SIGMOID])
agent = qa.QAgent(neural, 0.6, 0.99, 4)

def distance(x1, y1, x2, y2):
    return math.sqrt( ((x1 - x2)**2)+((y1 - y2)**2) )

def newState(state):
    dx = [1, 0, -1, 0, -1, -1, 1, 1]
    dy = [0, -1, 0, 1, 1, -1, 1, -1]
    response = [0] * 24
    x = head.xcor()
    y = head.ycor()
    values = convertCoords(y, x)
    xx = values[0]
    yy = values[1]
    for j in range(8):
        cx = xx + dx[j]
        cy = yy + dy[j]
        i = 1
        while not ((cx <= 0 or cy <= 0 or cx >= 30 or cy >= 30) or state[cx][cy] == 2 or state[cx][cy] == 1):
            i += 1
            cx = xx + dx[j] * i
            cy = yy + dy[j] * i
        if cx <= 0 or cy <= 0 or cx >= 30 or cy >= 30:
            response[j * 3 + 2] = 1 -distance(cx, cy, xx, yy) / 31
        if state[cx][cy] == 2:
            response[j * 3] = 1 - distance(cx, cy, xx, yy) / 31
        if state[cx][cy] == 1:
            response[j * 3 + 1] = 1 - distance(cx, cy, xx, yy) / 31
    return response

def move():
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)

    if head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)

    if head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)

    if head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)

def botState():
    global totalFrames
    stateNom = transState(state(width, height))
  #  newS = newState(state(width, height))
    action = 0
    action = agent.getBestActionWithRandomChance(stateNom, [], 0.15)
    if action == 0:
        go_up()

    if action == 1:
        go_down()

    if action == 2:
        go_left()

    if action == 3:
        go_right()
    totalFrames -= 1


# Keyboard bindings
wn.listen()
wn.onkeypress(go_up, "w")
wn.onkeypress(go_down, "s")
wn.onkeypress(go_left, "a")
wn.onkeypress(go_right, "d")

arb = []
arb.append((0, 100))
for index in range(1000):
    x = random.randint(-290, 290)
    y = random.randint(-290, 290)
    arb.append((x, y))

foodIndex = 0
# Main game loop
while True:
    wn.update()
    state(width, height, )
    # Check for a collision with the border
    if head.xcor()>290 or head.xcor()<-290 or head.ycor()>290 or head.ycor()<-290 or totalFrames == 0:
        time.sleep(1)
        head.goto(0,0)
        head.direction = "stop"
        foodIndex = 0
        food.goto(arb[foodIndex][0], arb[foodIndex][1])
        # Hide the segments
        for segment in segments:
            segment.goto(1000, 1000)

        # Clear the segments list
        segments.clear()
        print("Qeueue val ", score / 100.0 + (mex - totalFrames) / 10000)
        agent.trainTemporalDifferenceExpReplay(score / 100.0 + (mex - totalFrames) / 10000)
        # Reset the score
        score = 0

        # Reset the delay
        delay = speed
        totalFrames = mex

        pen.clear()
        pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))


    # Check for a collision with the food
    if head.distance(food) < 20:
        # Move the food to a random spot
     #   x = random.randint(-290, 290)
       # y = random.randint(-290, 290)
        food.goto(arb[foodIndex][0], arb[foodIndex][1])
        foodIndex += 1
        # Add a segment
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("grey")
        new_segment.penup()
        segments.append(new_segment)

        # Shorten the delay
        delay -= 0.001

        # Increase the score
        score += 10

        if score > high_score:
            high_score = score

        pen.clear()
        pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))

    # Move the end segments first in reverse order
    for index in range(len(segments)-1, 0, -1):
        x = segments[index-1].xcor()
        y = segments[index-1].ycor()
        segments[index].goto(x, y)

    # Move segment 0 to where the head is
    if len(segments) > 0:
        x = head.xcor()
        y = head.ycor()
        segments[0].goto(x,y)

    botState()
    move()

    # Check for head collision with the body segments
    for segment in segments:
        if segment.distance(head) < 20:
            time.sleep(1)
            head.goto(0,0)
            head.direction = "stop"
            foodIndex = 0
            food.goto(arb[foodIndex][0], arb[foodIndex][1])
            # Hide the segments
            for segment in segments:
                segment.goto(1000, 1000)

            # Clear the segments list
            segments.clear()

            # Reset the score
            print("Qeueue val ", score / 100.0 + (mex - totalFrames) / 10000)
            agent.trainTemporalDifferenceExpReplay(score / 100.0 + (mex - totalFrames) / 10000)
            score = 0
            totalFrames = mex

            # Reset the delay
            delay = speed

            # Update the score display
            pen.clear()
            pen.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Courier", 24, "normal"))

    time.sleep(delay)

wn.mainloop()