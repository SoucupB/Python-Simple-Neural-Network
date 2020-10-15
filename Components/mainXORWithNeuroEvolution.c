#include <stdio.h>
#include "NeuralNetwork.h"
#include "NeuroEvolution.h"
#include <time.h>
#include <math.h>
#include "Functions.h"
#include <stdlib.h>
#include "ExperienceReplay.h"

int main() {
    // XOR problem with neuroevolution!
    int32_t maxIterations = 60070;
    NeuroBatch batch;
    int32_t inputs[] = {2, 2, 1};
    float input[5][5] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float output[5][5] = {{0}, {1}, {1}, {0}};
    int32_t maxNets = 30;
    int32_t functionsIndex[] = {RELU, SIGMOID};
    // ExperienceReplay unit = er_Init();
    // float ana[] = {1, 0, 1}, response[] = {0};
    // float gana[] = {1, 0, 1, 5, 6, 2}, responser[] = {6, 5};
    // for(int32_t i = 0; i < 5; i++) {
    //     er_AddState(unit, ana, 3, response, 1);
    //     er_AddState(unit, gana, 6, responser, 2);
    // }
    // er_ShowStates(unit);
    // er_Clean(unit);
    // er_ShowStates(unit);
    // printf("DONE");
    // exit(0);
    batch = nb_Init(inputs, 3, maxNets, functionsIndex);
    printf("Started!\n");
    long mil = func_Time();
    for(int32_t i = 0; i < maxIterations; i++) {
        for(int32_t k = 0; k < maxNets; k++) {
            float maxFitnesPerNet = 300;
            for(int32_t j = 0; j < 4; j++) {
                float *netResponse = nb_NetworkResponse(batch, k, input[j], batch->configuration[0]);
             //   nn_Optimize(batch->nets[k], input[j], 2, output[j], 1);
             //   printf("%f ", netResponse[0]);
                maxFitnesPerNet -= fabs(output[j][0] - netResponse[0]) * 100;
                free(netResponse);
            }
            printf("Net %d has %.12f fitness!\n", k, maxFitnesPerNet);
            nb_AssignFitness(batch, k, maxFitnesPerNet / 300.0);
        }
        nb_CreateNewGeneration(batch, 0.01, 0.005);
    }
    printf("Done in: %ld miliseconds\n", func_Time() - mil);
    printf("Responses:\n");
    for(int32_t j = 0; j < 4; j++) {
        float *response = nn_FeedForward(batch->nets[0], input[j], 2);
        printf("Response for inputs [%.1f, %.1f] is %f\n", input[j][0], input[j][1], response[0]);
    }
    nb_Destroy(batch);
    printf("DONE");
}