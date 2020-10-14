#include "NeuroEvolution.h"
#include <stdlib.h>

NeuroBatch nb_Init(int32_t *configuration, int32_t confSize, int32_t numberOfNets, int32_t *functionsIndex) {
    func_UseSrand();
    NeuroBatch self = malloc(sizeof(struct NeuroBatch_t));
    self->nets = malloc(sizeof(NeuralNetwork) * numberOfNets);
    self->numberOfNets = numberOfNets;
    self->fitness = malloc(sizeof(float) * numberOfNets);
    for(int32_t i = 0; i < numberOfNets; i++) {
        self->fitness[i] = 0;
    }
    self->configuration = configuration;
    for(int32_t i = 0; i < numberOfNets; i++) {
        self->nets[i] = nn_InitMetaParameters(configuration, confSize, func_Uniform(MIN_LR_RANGE, MAX_LR_RANGE), functionsIndex);
    }
    return self;
}

void nb_AssignFitness(NeuroBatch batch, int32_t index, float fitness) {
    batch->fitness[index] = fitness;
}

NeuralNetwork nb_GetNet(NeuroBatch batch, int32_t index) {
    return batch->nets[index];
}

void nb_CreateNewGeneration(NeuroBatch batch, float chance, float by) {
    float *percentFitness = func_NormalizeArray(batch->fitness, batch->numberOfNets);
    int32_t indexes[140] = {0};
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        indexes[i] = i;
    }
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        for(int32_t j = i + 1; j < batch->numberOfNets; j++) {
            if(percentFitness[i] < percentFitness[j]) {
                float aux = percentFitness[i];
                percentFitness[i] = percentFitness[j];
                percentFitness[j] = aux;
                int32_t auxer = indexes[i];
                indexes[i] = indexes[j];
                indexes[j] = auxer;
            }
        }
    }
    NeuralNetwork *nets = malloc(sizeof(NeuralNetwork) * batch->numberOfNets);
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        int32_t netIndex = func_SelectFromProbabilities(percentFitness, batch->numberOfNets);
     //   int32_t max_index = rand() % 8;
        nets[i] = batch->nets[netIndex];
        nn_Mutate(nets[i], chance, by);
    }
    free(batch->nets);
    free(percentFitness);
    batch->nets = nets;
}

void nb_Destroy(NeuroBatch batch) {
    for(int32_t i = 0; i < batch->numberOfNets; i++) {
        nn_Destroy(batch->nets[i]);
    }
    free(batch->fitness);
    free(batch->configuration);
}