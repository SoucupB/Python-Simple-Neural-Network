#pragma once
#include <stdint.h>
#include "Neuron.h"
#include <assert.h>
#include "Functions.h"
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define IDENTITY 3
#define SOFTPLUS 4
#define ARCTAN 5
#define GAUSSIAN 6
#define MIN_INTERVAR -0.1
#define MAX_INTERVAR 0.1
#define FILE_NAME "networks.rt"

#define OPT_SGD 8
#define OPT_SGDM 9
#define OPT_SGDNM 10
#define OPT_ADAGRAD 11

#ifdef __cplusplus
extern "C" {
#endif

struct NeuralNetwork_t;
typedef struct NeuralNetwork_t* NeuralNetwork;

struct NeuralNetwork_t {
    Neuron *hiddens[1<<8];
    Neuron *biases;
    Neuron *inputs;
    Neuron *allNeurons;
    int32_t *hiddensSizes;
    int32_t numberOfHiddens;
    float lr;
    struct Function_t *functions;
    struct Function_t *dFunctions;
    int32_t maxNeurons;
    uint8_t train;
    int32_t hdSize;
    int32_t *structureBuffer;
    float ***matrixes;
    int32_t **prtStruct;
};


NeuralNetwork nn_InitMetaParameters(int32_t *structureBuffer, int32_t size, float lr, int32_t *configuration);
float *nn_FeedForward(NeuralNetwork net, float *structureBuffer);
void nn_ShowWeights(NeuralNetwork net);
float nn_Optimize(NeuralNetwork net, float *input, float *output, int8_t type);
void nn_ClearNeurons(NeuralNetwork net);
void nn_Destroy(NeuralNetwork net);
void nn_WriteFile(NeuralNetwork net, char *fileName);
void nn_LoadFile(NeuralNetwork network, char *fileNanme);
void nn_CrossOver(NeuralNetwork first, NeuralNetwork second);
void nn_Mutate(NeuralNetwork self, float chance, float by);
float elementFromBuffer(float *buffer, int32_t index);
int32_t nn_GetProportionalRandomIndex(NeuralNetwork net, float *input);

#ifdef __cplusplus
}
#endif
