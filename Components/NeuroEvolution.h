#pragma once
#include "NeuralNetwork.h"
#include "Functions.h"
#define MIN_LR_RANGE 0.0
#define MAX_LR_RANGE 0.25

struct NeuroBatch_t {
    NeuralNetwork *nets;
    float *fitness;
    int32_t numberOfNets;
    int32_t *configuration;
};

typedef struct NeuroBatch_t *NeuroBatch;

NeuroBatch nb_Init(int32_t *configuration, int32_t confSize, int32_t numberOfNets, int32_t *functionsIndex);
void nb_Destroy(NeuroBatch batch);
void nb_AssignFitness(NeuroBatch batch, int32_t index, float fitness);
NeuralNetwork nb_GetNet(NeuroBatch batch, int32_t index);
void nb_CreateNewGeneration(NeuroBatch batch, float chance, float by);