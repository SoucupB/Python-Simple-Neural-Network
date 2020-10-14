#include "ExperienceReplay.h"
#include <stdlib.h>
#include <stdio.h>

ExperienceReplay er_Init() {
    ExperienceReplay self = malloc(sizeof(struct ExperienceReplay_t));
    self->buffer = malloc(sizeof(float*));
    self->netResponse = malloc(sizeof(float*));
    self->bufferElementSize = malloc(sizeof(int32_t));
    self->valuesSizes = malloc(sizeof(int32_t));
    self->bufferSize = 0;
    self->capacity = 1;
    return self;
}

void er_AddState(ExperienceReplay self, float *state, int32_t stateSize, float *value, int32_t valueSize) {
    if(self->bufferSize >= self->capacity) {
        self->capacity *= 2;
        float **newBuffer = malloc(sizeof(float*) * self->capacity);
        float **newValue = malloc(sizeof(float*) * self->capacity);
        int32_t *statesSizes = malloc(sizeof(int32_t) * self->capacity);
        int32_t *valuesSizes = malloc(sizeof(int32_t) * self->capacity);
        for(int32_t i = 0; i < self->bufferSize; i++) {
            newBuffer[i] = self->buffer[i];
        }
        for(int32_t i = 0; i < self->bufferSize; i++) {
            newValue[i] = self->netResponse[i];
            statesSizes[i] = self->bufferElementSize[i];
            valuesSizes[i] = self->valuesSizes[i];
        }
        free(self->buffer);
        free(self->netResponse);
        free(self->bufferElementSize);
        free(self->valuesSizes);
        self->buffer = newBuffer;
        self->netResponse = newValue;
        self->bufferElementSize = statesSizes;
        self->valuesSizes = valuesSizes;
    }
    float *copyBuffer = malloc(sizeof(float) * stateSize);
    for(int32_t i = 0; i < stateSize; i++) {
        copyBuffer[i] = state[i];
    }
    float *copyValues = malloc(sizeof(float) * valueSize);
    for(int32_t i = 0; i < valueSize; i++) {
        copyValues[i] = value[i];
    }
    self->buffer[self->bufferSize] = copyBuffer;
    self->netResponse[self->bufferSize] = copyValues;
    self->bufferElementSize[self->bufferSize] = stateSize;
    self->valuesSizes[self->bufferSize] = valueSize;
    self->bufferSize++;
}

void er_ShowStates(ExperienceReplay self) {
    printf("\nStates\n");
    for(int32_t i = 0; i < self->bufferSize; i++) {
        for(int32_t j = 0; j < self->bufferElementSize[i]; j++) {
            printf("%.1f ", self->buffer[i][j]);
        }
        printf("\n");
    }
    printf("\nValues\n");
    for(int32_t i = 0; i < self->bufferSize; i++) {
        for(int32_t j = 0; j < self->valuesSizes[i]; j++) {
            printf("%.3f ", self->netResponse[i][j]);
        }
        printf("\n");
    }
}

void er_Clean(ExperienceReplay self) {
    for(int32_t i = 0; i < self->bufferSize; i++) {
        free(self->buffer[i]);
        free(self->netResponse[i]);
        self->buffer[i] = NULL;
        self->netResponse[i] = NULL;
        self->bufferElementSize[i] = 0;
        self->valuesSizes[i] = 0;
    }
    self->bufferSize = 0;
}

void er_Destroy(ExperienceReplay self) {
    for(int32_t i = 0; i < self->bufferSize; i++) {
        free(self->buffer[i]);
        free(self->netResponse[i]);
        self->buffer[i] = NULL;
        self->netResponse[i] = NULL;
        self->bufferElementSize[i] = 0;
        self->valuesSizes[i] = 0;
    }
    free(self->buffer);
    free(self->netResponse);
    free(self->bufferElementSize);
    free(self->valuesSizes);
    free(self);
}

float *er_GetValue(ExperienceReplay self, int32_t index) {
    return self->netResponse[index];
}

float *er_GetState(ExperienceReplay self, int32_t index) {
    return self->buffer[index];
}