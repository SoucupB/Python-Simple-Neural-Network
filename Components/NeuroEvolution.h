#pragma once
#include "NeuralNetwork.h"
#include "Functions.h"
#include "ExperienceReplay.h"
#define MIN_LR_RANGE 0.0
#define MAX_LR_RANGE 0.25

struct NeuroBatch_t {
    NeuralNetwork *nets;
    float *fitness;
    int32_t numberOfNets;
    int32_t *configuration;
    ExperienceReplay *replays;
};

typedef struct NeuroBatch_t *NeuroBatch;

NeuroBatch nb_Init(int32_t *configuration, int32_t confSize, int32_t numberOfNets, int32_t *functionsIndex);
void nb_Destroy(NeuroBatch batch);
void nb_AssignFitness(NeuroBatch batch, int32_t index, float fitness);
NeuralNetwork nb_GetNet(NeuroBatch batch, int32_t index);
void nb_CreateNewGeneration(NeuroBatch batch, float chance, float by);
float *nb_NetworkResponse(NeuroBatch self, int32_t index, float *state, int32_t size);
void nb_MutateGradientDescent(NeuroBatch self, int32_t index, float fitness);