# Python-Simple-Neural-Network
It is a NeuralNetwork library build with a C backend, great for classification problems and it has some reinforcement learning agents good for games (examples below).
<br>
Its very easy to use, (no external librarys needed, not even numpy, just the build in 'math' and 'random' for python).
<br>
The C backend is using built in librarys like stdio, math etc... and a hashmap library from this repository.

<br>
Requirements!
<br>
This version is supported by any Python 3.x x64 version.
<br>
GCC (x86_64-posix-seh-rev0, Built by MinGW-W64 project) > 8.1.0 (It might work with other types of GCC aswell).
<br>
<br>
While it might be a python library it can be used in C aswell (See MainXOR.c for the XOR problem solved with the neural network).
Here is an example of TICTACTOE bot implemented with the help of this library and its Q learning methods (See tictacQLtes.py).
Before everything, the C library should be compiled with the command!
<code>
gcc -fPIC -shared NeuralNetwork.c hashmap.c Functions.c Neuron.c QAgent.c -Wall -o NeuralNetwork.so -O9
</code>
This will create a shared binary library in order for python to make is work!
<br>
To test it in C exclusively use this command!
<code>
gcc NeuralNetwork.c hashmap.c Functions.c Neuron.c MainXOR.c QAgent.c -o application -Wall -O9 -lm
</code>
Then run application with!
<code>
./application
</code>

To run this example run
<code>
python tictacQLtest.py
</code>
![alt text](Plots/TicTacToe_wins.png)

