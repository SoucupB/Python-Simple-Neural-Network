#include "QAgent.h"

QAgent qa_Init(NeuralNetwork brain, float lr, float discount, int32_t numberOfActions) {
    QAgent agent = malloc(sizeof(struct QAgent_t));
    agent->brain = brain;
    agent->lr = lr;
    agent->discount = discount;
    agent->numberOfActions = numberOfActions;
    agent->stateSize = brain->hiddensSizes[0] - numberOfActions;
    return agent;
}

int32_t qa_GetChoosenActionIndex(QAgent self, float *state, int32_t *prohibitedActions, int32_t prhSize) {
    float maxQValue = -1e9;
    int32_t actionIndex = 0;
    int32_t prohibitedActionsCounters[MAX_STATE_SIZE] = {0};
    float stateMixedWithAction[MAX_ACTIONS_NUMBER] = {0};
    for(int32_t i = 0; i < prhSize; i++) {
        prohibitedActionsCounters[prohibitedActions[i]] = 1;
    }
    for(int32_t i = 0; i < self->stateSize; i++) {
        stateMixedWithAction[i] = state[i];
    }
    int32_t inputSize = self->numberOfActions + self->stateSize;
    for(int32_t i = 0; i < self->numberOfActions; i++) {
        if(!prohibitedActionsCounters[i]) {
            stateMixedWithAction[i + self->stateSize] = 1;
            float *currentQValue = nn_FeedForward(self->brain, stateMixedWithAction, inputSize);
            if(currentQValue[0] > maxQValue) {
                maxQValue = currentQValue[0];
                actionIndex = i;
            }
            free(currentQValue);
            stateMixedWithAction[i + self->stateSize] = 0;
        }
    }
    return actionIndex;
}

void qa_TrainTemporalDifference(QAgent self, float **inputBuffer, int32_t *actionIndex, float endStateReward, int32_t size) {
    float *result = malloc(sizeof(float) * size);
    result[size - 1] = endStateReward;
    float stateMixedWithAction[MAX_ACTIONS_NUMBER] = {0};
    for(int32_t i = size - 2; i >= 0; i--) {
        for(int32_t j = 0; j < self->stateSize; j++) {
            stateMixedWithAction[j] = inputBuffer[i][j];
        }
        stateMixedWithAction[actionIndex[i] + self->stateSize] = 1;
        float *netGuess = nn_FeedForward(self->brain, inputBuffer[i], self->stateSize);
        result[i] = netGuess[0] + self->lr * (result[i + 1] - netGuess[0]);
        free(netGuess);
    }
    for(int32_t i = size - 1; i >= 0; i--) {
        for(int32_t j = 0; j < self->stateSize; j++) {
            stateMixedWithAction[j] = inputBuffer[i][j];
        }
        stateMixedWithAction[actionIndex[i] + self->stateSize] = 1;
        float resultVector[] = {result[i]};
        nn_Optimize(self->brain, stateMixedWithAction, self->numberOfActions + self->stateSize, resultVector, 1);
    }
}

void qa_Destroy(QAgent self) {
    free(self);
}