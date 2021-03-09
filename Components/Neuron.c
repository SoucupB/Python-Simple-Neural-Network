#include "Neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Functions.h"
#include "StaticAllocator.h"

//static inline void saveWeightMatrix(Neuron self, int32_t parentID, int32_t childID, float value);
//static inline float getWeightMatrix(Neuron self, int32_t parentID, int32_t childID);

Neuron ne_Init(int32_t ID,
               float (*activationFunction)(float), float (*derivativeActivationFunction)(float), float lr) {
    Neuron neuron = nmalloc(sizeof(struct Neuron_t));
    neuron->childs = vct_Init(sizeof(Neuron));
    neuron->parents = vct_Init(sizeof(Neuron));
    neuron->value = 0;
    neuron->error = 0;
    neuron->ID = ID;
    neuron->unChangedValue = 0;
    neuron->shouldApplyActivation = 1;
    neuron->activationFunction = activationFunction;
    neuron->derivativeActivationFunction = derivativeActivationFunction;
    neuron->lr = lr;
    neuron->pastGradient = 0;
    neuron->epsilon = 1e-5;
    neuron->beta = 0.9;
    neuron->matrix = NULL;
    neuron->matrixIndexes = NULL;
    return neuron;
}

void ne_AddMatrix(Neuron self, float ***matrix, int32_t **indexes) {
    self->matrix = matrix;
    self->matrixIndexes = indexes;
}

void ne_Destroy(Neuron neuron) {
    vct_Delete(neuron->childs);
    vct_Delete(neuron->parents);
    nfree(neuron);
}

void saveWeightMatrix(Neuron self, int32_t parentID, int32_t childID, float value) {
    int32_t matrIndex = smin(self->matrixIndexes[1][parentID], self->matrixIndexes[1][childID]);
    self->matrix[matrIndex][self->matrixIndexes[0][parentID]][self->matrixIndexes[0][childID]] = value;
}

float getWeightMatrix(Neuron self, int32_t parentID, int32_t childID) {
    int32_t matrIndex = smin(self->matrixIndexes[1][parentID], self->matrixIndexes[1][childID]);
    return self->matrix[matrIndex][self->matrixIndexes[0][parentID]][self->matrixIndexes[0][childID]];
}

void ne_Tie(Neuron parent, Neuron child, float value) {
    vct_Push(parent->childs, &child);
    vct_Push(child->parents, &parent);
    saveWeightMatrix(parent, parent->ID, child->ID, value);
}

void ne_FeedForward(Neuron neuron) {
    if(neuron->shouldApplyActivation) {
        ne_Activate(neuron);
    }
    for(int32_t i = 0; i < neuron->childs->size; i++) {
        Neuron child = ((Neuron *)neuron->childs->buffer)[i];
        child->value += neuron->value * getWeightMatrix(neuron, neuron->ID, child->ID);
        child->unChangedValue = child->value;
    }
}

void ne_OptimizeSGD(Neuron self) {
    for(int32_t i = 0; i < self->parents->size; i++) {
        Neuron parent = ((Neuron *)self->parents->buffer)[i];
        float gradient = getWeightMatrix(self, parent->ID, self->ID) +
                         self->error * self->lr * self->derivativeActivationFunction(self->unChangedValue) * parent->value;
        saveWeightMatrix(parent, parent->ID, self->ID, gradient);
    }
}

void ne_Activate(Neuron neuron) {
    assert(neuron->activationFunction != NULL);
    neuron->value = neuron->activationFunction(neuron->value);
}

void ne_PropagateErrorToParents(Neuron self) {
    for(int32_t i = 0; i < self->parents->size; i++) {
        Neuron parent = ((Neuron *)self->parents->buffer)[i];
        parent->error += self->error * getWeightMatrix(self, parent->ID, self->ID);
    }
}

void ne_OptimizeAdagrad(Neuron self) {
    for(int32_t i = 0; i < self->parents->size; i++) {
        Neuron parent = ((Neuron *)self->parents->buffer)[i];
        float gradient = self->error * self->derivativeActivationFunction(self->unChangedValue) * parent->value;
        self->pastGradient += gradient * gradient;
        float lastWeight = getWeightMatrix(parent, parent->ID, self->ID);
        lastWeight = lastWeight + self->lr / (sqrt(self->pastGradient + self->epsilon)) * gradient;
        saveWeightMatrix(parent, parent->ID, self->ID, lastWeight);
    }
}

void ne_OptimizeSgdMomentum(Neuron self) {
    for(int32_t i = 0; i < self->parents->size; i++) {
        Neuron parent = ((Neuron *)self->parents->buffer)[i];
        float lastWeight = getWeightMatrix(parent, parent->ID, self->ID);
        float gradient = self->derivativeActivationFunction(self->unChangedValue) * parent->value * self->error;
        self->pastGradient = self->beta * self->pastGradient + self->lr * gradient;
        saveWeightMatrix(parent, parent->ID, self->ID, lastWeight + self->pastGradient);
    }
}

void ne_OptimizeSgdNesterovMomentum(Neuron self) {
    for(int32_t i = 0; i < self->parents->size; i++) {
        Neuron parent = ((Neuron *)self->parents->buffer)[i];
        float lastWeight = getWeightMatrix(parent, parent->ID, self->ID);
        float gradient = self->derivativeActivationFunction(self->unChangedValue) * parent->value * self->error;
        self->pastGradient = self->beta * self->pastGradient + self->lr * gradient;
        saveWeightMatrix(parent, parent->ID, self->ID, lastWeight + self->pastGradient);
    }
}

void ne_NesterovFeedForward(Neuron self) {
    if(self->shouldApplyActivation) {
        ne_Activate(self);
    }
    for(int32_t i = 0; i < self->childs->size; i++) {
        Neuron child = ((Neuron *)self->childs->buffer)[i];
        child->value += self->value * (getWeightMatrix(self, self->ID, child->ID) - self->beta * self->pastGradient);
        child->unChangedValue = child->value;
    }
}