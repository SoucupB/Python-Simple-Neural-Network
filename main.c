#include <stdio.h>
#include "NeuralNetwork.h"
#include <time.h>
#include <math.h>

long millis(){
    struct timespec _t;
    clock_gettime(CLOCK_REALTIME, &_t);
    return _t.tv_sec*1000 + lround(_t.tv_nsec/1.0e6);
}


int main() {
    NeuralNetwork network;
    int32_t inputs[] = {2, 2, 1}; // do
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    int32_t functionsIndex[] = {SIGMOID, SIGMOID}; // do
    network = nn_InitMetaParameters(inputs, 3, 0.1, functionsIndex);
    //nn_ShowWeights(network);
    printf("Started!\n");
    long mil = millis();
    for(int32_t i = 0; i < 70000; i++) {
        for(int32_t j = 0; j < 4; j++) {
            printf("%f\n", nn_Optimize(network, input[j], 2, output[j], 1)); // do
        }
    }
    printf("Done in: %ld miliseconds\n", millis() - mil);
    printf("Responses:\n");
    for(int32_t j = 0; j < 4; j++) {
        float *response = nn_FeedForward(network, input[j], 2);
        printf("%f\n", response[0]);
    }
   // nn_WriteFile(network);
    nn_Destroy(network);
    printf("DONE");
}