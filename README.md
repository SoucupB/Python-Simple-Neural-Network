# Python-Simple-Neural-Network
It is a NeuralNetwork library build with a C backend, great for classification problems and it has some reinforcement learning agents good for games (examples below).
Its very easy to use, (no external librarys needed, not even numpy, just the build in 'math' and 'random' for python).
The C backend is using built in librarys like stdio, math, etc....

# Requirements

    python 3.x version!

    GCC (x86_64-posix-seh-rev0, Built by MinGW-W64 project) > 8.1.0 (It might work with other types of GCC aswell).

While it might be a python library it can be used in C aswell (See MainXOR.c for the XOR problem solved with the neural network).

# Compilation
This will compile the C backend library for the python interface

    gcc -fPIC -shared Components/NeuralNetwork.c Components/Vector.c Components/StaticAllocator.c Components/NeuroEvolution.c Components/Functions.c Components/Neuron.c Components/QAgent.c Components/ExperienceReplay.c -Wall -o  Components/NeuralNetwork.so -O9

To create and test it in C exclusively use this command!

    gcc Components/NeuralNetwork.c Components/Vector.c Components/StaticAllocator.c Components/NeuroEvolution.c Components/Functions.c Components/Neuron.c Components/mainXORWithNeuroEvolution.c Components/QAgent.c Components/ExperienceReplay.c -o application -Wall -O9 -lm

    gcc Components/Vector.c Components/StaticAllocator.c Components/NeuralNetwork.c Components/NeuroEvolution.c Components/Functions.c Components/Neuron.c Components/MainXOR.c Components/QAgent.c Components/ExperienceReplay.c -o application -Wall -O9

Then run application in C only!

    ./application

# Libs
While a big lib is not necessary, to run the statistical builder you will need matplotlib for it in order to make it work.
This lib can be optained with

    pip install matplotlib

# Components
This library is made from 2 components.
# Neural Network
This is the component for the function aproximator used for aproximation of the Q table.
It can also be used by other purpuses as well, for image recognition and other stuff!.
# QAgent
QAgent is the component used for a game agent, see tictacQLtest.py for the tictactoe example.

# Examples
Here is a graph that portray the evolution of the Q agent against a random agent.
The agent is playing 350 batches of 1000 games each! (It plays as the 'O' player and random player plays as 'X')

To run this example run

    python tictacQLtest.py

![alt text](Plots/TicTacToe_wins.png)

