#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include <stdlib.h>

int main() {
    // XOR problem!
    int32_t maxIterations = 70000;
    NeuralNetwork network;
    int32_t inputs[] = {2, 4, 1};
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    int32_t functionsIndex[] = {RELU, SIGMOID};
    network = nn_InitMetaParameters(inputs, 3, 0.1, functionsIndex);
    // for(int32_t i = 0; i < 100000; i++)
    //     nn_Mutate(network, 0.1, 0.01);
    // exit(0);
    printf("Started!\n");
    long mil = func_Time();
    for(int32_t i = 0; i < maxIterations; i++) {
        for(int32_t j = 0; j < 4; j++) {
            nn_Optimize(network, input[j], 2, output[j], 1);
        }
    }
    printf("Done in: %ld miliseconds\n", func_Time() - mil);
    printf("Responses:\n");
    for(int32_t j = 0; j < 4; j++) {
        float *response = nn_FeedForward(network, input[j], 2);
        printf("Response for inputs [%.1f, %.1f] is %f\n", input[j][0], input[j][1], response[0]);
    }
    nn_Destroy(network);
    printf("DONE");
}