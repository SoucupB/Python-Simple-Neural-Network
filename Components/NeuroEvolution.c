#include "NeuroEvolution.h"
#include <stdlib.h>

NeuroBatch nb_Init(int32_t *configuration, int32_t confSize, int32_t numberOfNets, int32_t *functionsIndex) {
    func_UseSrand();
    NeuroBatch self = malloc(sizeof(struct NeuroBatch_t));
    self->nets = malloc(sizeof(NeuralNetwork) * numberOfNets);
    self->replays = malloc(sizeof(ExperienceReplay) * numberOfNets);
    self->numberOfNets = numberOfNets;
    self->fitness = malloc(sizeof(float) * numberOfNets);
    for(int32_t i = 0; i < numberOfNets; i++) {
        self->fitness[i] = 0;
    }
    self->configuration = configuration;
    for(int32_t i = 0; i < numberOfNets; i++) {
        self->nets[i] = nn_InitMetaParameters(configuration, confSize, func_Uniform(MIN_LR_RANGE, MAX_LR_RANGE), functionsIndex);
        self->replays[i] = er_Init();
    }
    return self;
}

void nb_AssignFitness(NeuroBatch batch, int32_t index, float fitness) {
    batch->fitness[index] = fitness;
}

NeuralNetwork nb_GetNet(NeuroBatch batch, int32_t index) {
    return batch->nets[index];
}

float *nb_NetworkResponse(NeuroBatch self, int32_t index, float *state, int32_t size) {
    float *net_Response = nn_FeedForward(self->nets[index], state, size);
    int32_t netsSizes = self->nets[index]->hiddensSizes[self->nets[index]->numberOfHiddens];
    er_AddState(self->replays[index], state, size, net_Response, netsSizes);
    return net_Response;
}

void nb_CreateNewGeneration(NeuroBatch batch, float chance, float by) {
    float *percentFitness = func_NormalizeArray(batch->fitness, batch->numberOfNets);
    NeuralNetwork *nets = malloc(sizeof(NeuralNetwork) * batch->numberOfNets);
    float *fitness = malloc(sizeof(float) * batch->numberOfNets);
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        int32_t netIndex = func_SelectFromProbabilities(percentFitness, batch->numberOfNets);
        nets[i] = batch->nets[netIndex];
        fitness[i] = batch->fitness[netIndex];
    }
    free(batch->nets);
    free(percentFitness);
    free(batch->fitness);
    batch->fitness = fitness;
    batch->nets = nets;
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        nb_MutateGradientDescent(batch, i, batch->fitness[i]);
    }
}

void nb_Destroy(NeuroBatch batch) {
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        nn_Destroy(batch->nets[i]);
        er_Destroy(batch->replays[i]);
    }
    free(batch->fitness);
    free(batch->configuration);
}

void nb_MutateGradientDescent(NeuroBatch self, int32_t index, float fitness) {
    int32_t outputSize = self->nets[index]->hiddensSizes[self->nets[index]->numberOfHiddens];
    for(int32_t i = 0; i < self->replays[index]->bufferSize; i++) {
        float *outputRandomTarget = malloc(sizeof(float) * outputSize);
        for(int32_t j = 0; j < outputSize; j++) {
            outputRandomTarget[j] = er_GetValue(self->replays[index], i)[j] + func_RandomNumber(-0.1, 0.1) * (1.0 - fitness) * (1.0 - fitness);
        }
        nn_Optimize(self->nets[index], er_GetState(self->replays[index], i), self->nets[index]->hiddensSizes[0], outputRandomTarget, outputSize, OPT_SGD);
        free(outputRandomTarget);
    }
    er_Clean(self->replays[index]);
}