#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include <stdlib.h>

int main() {
    // XOR problem!
    func_UseSrand();
    int32_t maxIterations = 50408;
    NeuralNetwork network;
    int32_t inputs[] = {2, 4, 1};
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    int32_t functionsIndex[] = {RELU, SIGMOID};
    network = nn_InitMetaParameters(inputs, 3, 0.1, functionsIndex);
    printf("Started!\n");
    long mil = func_Time();
    for(int32_t i = 0; i < maxIterations; i++) {
        for(int32_t j = 0; j < 4; j++) {
            nn_Optimize(network, input[j], output[j], OPT_SGD);
        }
    }
    printf("Done in: %ld miliseconds\n", func_Time() - mil);
    printf("Responses:\n");
    for(int32_t j = 0; j < 4; j++) {
        float *response = nn_FeedForward(network, input[j]);
        printf("Response for inputs [%.1f, %.1f] is %f\n", input[j][0], input[j][1], response[0]);
    }
    nn_WriteFile(network, "inputload.co");
    nn_Destroy(network);
    printf("DONE");
}