#pragma once
#include <stdint.h>
#include "hashmap.h"
#define MAX_CONNECTIONS 512

struct Neuron_t;
typedef struct Neuron_t *Neuron;

struct Neuron_t {
    int32_t ID;
    Neuron *childs; // no more then 512 conections per nodes.
    Neuron *parents;
    int32_t childsCount;
    int32_t parentsCount;
    struct hashmap *hash;
    float value;
    float error;
    float unChangedValue;
    int8_t shouldApplyActivation;
    float (*activationFunction)(float);
    float (*derivativeActivationFunction)(float);
    float lr;
};

Neuron ne_Init(int32_t ID, struct hashmap *hash, float (*activationFunction)(float), float (*derivativeActivationFunction)(float), float lr);
void ne_FeedForward(Neuron neuron);
void ne_Optimize(Neuron neuron);
void ne_Tie(Neuron parent, Neuron child, float value);
void ne_OptimizeSGD(Neuron neuron);
void ne_Activate(Neuron neuron);
void ne_PropagateErrorToParents(Neuron neuron);
void ne_Destroy(Neuron neuron);