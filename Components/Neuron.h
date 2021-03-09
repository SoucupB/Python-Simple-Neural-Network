#pragma once
#include <stdint.h>
#include "Vector.h"
#define MAX_CONNECTIONS 2048

struct Neuron_t;
typedef struct Neuron_t *Neuron;

struct Neuron_t {
    int32_t ID;
    Vector childs;
    Vector parents;
    float value;
    float error;
    float unChangedValue;
    int8_t shouldApplyActivation;
    float (*activationFunction)(float);
    float (*derivativeActivationFunction)(float);
    float lr;
    float pastGradient;
    float epsilon;
    float beta;
    float ***matrix;
    int32_t **matrixIndexes;
};

Neuron ne_Init(int32_t ID, float (*activationFunction)(float), float (*derivativeActivationFunction)(float), float lr);
void ne_FeedForward(Neuron neuron);
void ne_Optimize(Neuron neuron);
void ne_Tie(Neuron parent, Neuron child, float value);
void ne_OptimizeSGD(Neuron neuron);
void ne_Activate(Neuron neuron);
void ne_PropagateErrorToParents(Neuron neuron);
void ne_Destroy(Neuron neuron);
void ne_OptimizeSgdMomentum(Neuron self);
void ne_OptimizeSgdNesterovMomentum(Neuron self);
void ne_NesterovFeedForward(Neuron self);
void ne_OptimizeAdagrad(Neuron self);
void ne_AddMatrix(Neuron self, float ***matrix, int32_t **indexes);
float getWeightMatrix(Neuron self, int32_t parentID, int32_t childID);
void saveWeightMatrix(Neuron self, int32_t parentID, int32_t childID, float value);