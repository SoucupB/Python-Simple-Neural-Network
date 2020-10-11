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

float getQValueFromState(QAgent self, float *state, int32_t action) {
    float stateMixedWithAction[MAX_ACTIONS_NUMBER] = {0};
    for(int32_t j = 0; j < self->stateSize; j++) {
        stateMixedWithAction[j] = state[j];
    }
    stateMixedWithAction[action + self->stateSize] = 1;
    float *qValueBuffer = nn_FeedForward(self->brain, stateMixedWithAction, self->stateSize + self->numberOfActions);
    float qValue = qValueBuffer[0];
    free(qValueBuffer);
    return qValue;
}

void optimizeQValue(QAgent self, float *state, int32_t action, float value) {
    float stateMixedWithAction[MAX_ACTIONS_NUMBER] = {0};
    for(int32_t j = 0; j < self->stateSize; j++) {
        stateMixedWithAction[j] = state[j];
    }
    float resultVector[] = {value};
    stateMixedWithAction[action + self->stateSize] = 1;
    nn_Optimize(self->brain, stateMixedWithAction, self->stateSize + self->numberOfActions, resultVector, 1);
}

void qa_TrainTemporalDifference(QAgent self, float **inputBuffer, int32_t *actionIndex, float endStateReward, int32_t size) {
    float *result = malloc(sizeof(float) * size);
    result[size - 1] = endStateReward;
    for(int32_t i = size - 2; i >= 0; i--) {
        float qValue = getQValueFromState(self, inputBuffer[i], actionIndex[i]);
        result[i] = qValue + self->lr * (result[i + 1] - qValue);
    }
    for(int32_t i = size - 1; i >= 0; i--) {
        optimizeQValue(self, inputBuffer[i], actionIndex[i], result[i]);
    }
    free(result);
}

float getMaxQValue(QAgent self, float *currentState) {
    float maximum = -1e9;
    int32_t actionsNumber = self->numberOfActions;
    for(int32_t i = 0; i < actionsNumber; i++) {
        float qValue = getQValueFromState(self, currentState, i);
        if(qValue > maximum) {
            maximum = qValue;
        }
    }
    return maximum;
}

void qa_TrainDeepQNet(QAgent self, float **inputBuffer, int32_t *actionIndex, float *rewards, int32_t size) {
    float *discounterReward = malloc(sizeof(float) * size);
    float *qDelta = malloc(sizeof(float) * size);
    float discount = self->discount;
    discounterReward[size - 1] = rewards[size - 1];
    qDelta[size - 1] = discounterReward[size - 1];
    for(int32_t i = size - 2; i >= 0; i--) {
        discounterReward[i] = discounterReward[i + 1] + rewards[i] * discount;
        discount *= self->discount;
    }
    for(int32_t i = size - 2; i >= 0; i--) {
        float maxQValueNextState = getMaxQValue(self, inputBuffer[i + 1]);
        float qValue = getQValueFromState(self, inputBuffer[i], actionIndex[i]);
        float currentQValue = qValue + self->lr * (discounterReward[i] + maxQValueNextState - qValue);
        qDelta[i] = currentQValue;
    }
    for(int32_t i = size - 1; i >= 0; i--) {
        optimizeQValue(self, inputBuffer[i], actionIndex[i], qDelta[i]);
    }
    free(discounterReward);
    free(qDelta);
}

void qa_Destroy(QAgent self) {
    free(self);
}