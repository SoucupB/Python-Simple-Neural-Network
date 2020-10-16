#pragma once
#include "NeuralNetwork.h"
#include <stdlib.h>
#include "ExperienceReplay.h"
#define MAX_STATE_SIZE (1<<11)
#define MAX_ACTIONS_NUMBER (1<<11)

struct QAgent_t {
    NeuralNetwork brain;
    float lr;
    float discount;
    int32_t numberOfActions;
    int32_t stateSize;
    ExperienceReplay replay;
};

typedef struct QAgent_t *QAgent;

QAgent qa_Init(NeuralNetwork brain, float lr, float discount, int32_t numberOfActions);
void qa_Destroy(QAgent self);
void qa_TrainTemporalDifference(QAgent self, float **inputBuffer, int32_t *actionIndex, float endStateReward, int32_t size);
void qa_TrainDeepQNet(QAgent self, float **inputBuffer, int32_t *actionIndex, float *rewards, int32_t size);
int32_t qa_GetChoosenActionIndex(QAgent self, float *state, int32_t *prohibitedActions, int32_t prhSize);
int32_t qa_GetActionWithRandom(QAgent self, float *state, int32_t *prohibitedActions, int32_t prhSize, float chance);
void qa_ShowExperienceReplay(QAgent agent);
void qa_TrainTemporalDifferenceReplay(QAgent self, float endStateReward, int8_t type);
void qa_TrainTemporalDifference(QAgent self, float **inputBuffer, int32_t *actionIndex, float endStateReward, int32_t size);