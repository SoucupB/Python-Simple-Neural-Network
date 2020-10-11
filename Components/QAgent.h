#pragma once
#include "NeuralNetwork.h"
#include <stdlib.h>
#define MAX_STATE_SIZE 1<<9
#define MAX_ACTIONS_NUMBER 1<<9

struct QAgent_t {
    NeuralNetwork brain;
    float lr;
    float discount;
    int32_t numberOfActions;
    int32_t stateSize;
};

typedef struct QAgent_t *QAgent;

QAgent qa_Init(NeuralNetwork brain, float lr, float discount, int32_t numberOfActions);
void qa_Destroy(QAgent self);
void qa_TrainTemporalDifference(QAgent self, float **inputBuffer, int32_t *actionIndex, float endStateReward, int32_t size);
void qa_TrainDeepQNet(QAgent self, float **inputBuffer, int32_t *actionIndex, float *rewards, int32_t size);
int32_t qa_GetChoosenActionIndex(QAgent self, float *state, int32_t *prohibitedActions, int32_t prhSize);