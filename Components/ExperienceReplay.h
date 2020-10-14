#pragma once
#include <stdint.h>

struct ExperienceReplay_t {
    float **buffer;
    float **netResponse;
    int32_t bufferSize;
    int32_t capacity;
    int32_t *bufferElementSize;
    int32_t *valuesSizes;
};

typedef struct ExperienceReplay_t *ExperienceReplay;

ExperienceReplay er_Init();
void er_Destroy(ExperienceReplay self);
void er_AddState(ExperienceReplay self, float *state, int32_t stateSize, float *value, int32_t valueSize);
float *er_GetState(ExperienceReplay self, int32_t index);
float *er_GetValue(ExperienceReplay self, int32_t index);
void er_ShowStates(ExperienceReplay self);
void er_Clean(ExperienceReplay self);