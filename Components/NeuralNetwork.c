#include "NeuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "StaticAllocator.h"
#include <string.h>

float *feedForwardTrain(NeuralNetwork net, float *structureBuffer);

float **makeMatrix(int32_t sizeY, int32_t sizeX) {
    float **matrix = nmalloc(sizeof(float **) * sizeY);
    for(int32_t i = 0; i < sizeY; i++) {
        float *row = nmalloc(sizeof(float) * sizeX);
        memset(row, 0, sizeof(float) * sizeX);
        matrix[i] = row;
    }
    return matrix;
}

void showDataMatrix(NeuralNetwork net);

NeuralNetwork nn_InitMetaParameters(int32_t *structureBuffer, int32_t size, float lr, int32_t *configuration) {
    NeuralNetwork neuralNetwork = block_Alloc(sizeof(struct NeuralNetwork_t));
    block_SwitchStatus(neuralNetwork);
    int32_t *hiddensSizes = nmalloc(sizeof(int32_t) * size);
    neuralNetwork->hiddensSizes = hiddensSizes;
    neuralNetwork->structureBuffer = nmalloc(sizeof(int32_t) * size);
    memcpy(neuralNetwork->structureBuffer, structureBuffer, sizeof(int32_t) * size);
    neuralNetwork->hdSize = size;
    struct Function_t *functions = func_GetActivationFunctions();
    struct Function_t *dFunctions = func_GetDActivationFunctions();
    neuralNetwork->lr = lr;
    neuralNetwork->matrixes = nmalloc(sizeof(float **) * size);
    for(int32_t i = 0; i < size - 1; i++) {
        neuralNetwork->matrixes[i] = makeMatrix(structureBuffer[i] + 1, structureBuffer[i + 1] + 1);
    }
    int32_t **prtStruct = nmalloc(sizeof(int32_t *) * 2);
    int32_t *parents = nmalloc((func_ArraySum(structureBuffer, size) + size) * sizeof(int32_t));
    int32_t *matrIndexes = nmalloc((func_ArraySum(structureBuffer, size) + size) * sizeof(int32_t));
    neuralNetwork->prtStruct = prtStruct;
    prtStruct[0] = parents;
    prtStruct[1] = matrIndexes;
    neuralNetwork->numberOfHiddens = 0;
    neuralNetwork->functions = functions;
    neuralNetwork->dFunctions = dFunctions;
    neuralNetwork->train = 0;
    int32_t ids = 0;
    stat_Init(neuralNetwork, structureBuffer[size - 1]);
    Neuron *layer = nmalloc(sizeof(Neuron) * structureBuffer[0]);
    Neuron *allNeurons = nmalloc(sizeof(Neuron) * (func_ArraySum(structureBuffer, size) + size - 1));
    int32_t neuronsIndex = 0;
    neuralNetwork->biases = nmalloc(sizeof(struct Neuron_t) * size);
    for(int32_t j = 0; j < structureBuffer[0]; j++) {
        parents[ids] = j;
        matrIndexes[ids] = 0;
        layer[j] = ne_Init(ids++, NULL, NULL, neuralNetwork->lr);
        ne_AddMatrix(layer[j], neuralNetwork->matrixes, prtStruct);
        layer[j]->shouldApplyActivation = 0;
        allNeurons[neuronsIndex++] = layer[j];
    }
    neuralNetwork->hiddensSizes[0] = structureBuffer[0];
    neuralNetwork->inputs = layer;
    for(int32_t i = 1; i < size; i++) {
        layer = nmalloc(sizeof(Neuron) * structureBuffer[i]);
        for(int32_t j = 0; j < structureBuffer[i]; j++) {
            parents[ids] = j;
            matrIndexes[ids] = i;
            layer[j] = ne_Init(ids++, functions[configuration[i - 1]].func, dFunctions[configuration[i - 1]].func, neuralNetwork->lr);
            ne_AddMatrix(layer[j], neuralNetwork->matrixes, prtStruct);
            allNeurons[neuronsIndex++] = layer[j];
        }
        neuralNetwork->hiddens[neuralNetwork->numberOfHiddens++] = layer;
        neuralNetwork->hiddensSizes[i] = structureBuffer[i];
    }
    for(int32_t i = 0; i < neuralNetwork->hiddensSizes[0]; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[1]; j++) {
            ne_Tie(neuralNetwork->inputs[i], neuralNetwork->hiddens[0][j], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
        }
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens - 1; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[i + 1]; j++) {
            for(int32_t k = 0; k < neuralNetwork->hiddensSizes[i + 2]; k++) {
                ne_Tie(neuralNetwork->hiddens[i][j], neuralNetwork->hiddens[i + 1][k], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
            }
        }
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens; i++) {
        parents[ids] = structureBuffer[i];
        matrIndexes[ids] = i;
        neuralNetwork->biases[i] = ne_Init(ids++, functions[configuration[i]].func, dFunctions[configuration[i]].func, neuralNetwork->lr);
        ne_AddMatrix(neuralNetwork->biases[i], neuralNetwork->matrixes, prtStruct);
        neuralNetwork->biases[i]->value = 1.0;
        neuralNetwork->biases[i]->shouldApplyActivation = 0;
        allNeurons[neuronsIndex++] = neuralNetwork->biases[i];
    }
    for(int32_t i = 0; i < neuralNetwork->numberOfHiddens; i++) {
        for(int32_t j = 0; j < neuralNetwork->hiddensSizes[i + 1]; j++) {
            ne_Tie(neuralNetwork->biases[i], neuralNetwork->hiddens[i][j], func_Uniform(MIN_INTERVAR, MAX_INTERVAR));
        }
    }
    neuralNetwork->maxNeurons = ids;
    neuralNetwork->allNeurons = allNeurons;
    return neuralNetwork;
}

void free_Matrix(NeuralNetwork net) {
    for(int32_t i = 0; i < net->hdSize - 1; i++) {
        for(int32_t j = 0; j < net->structureBuffer[i]; j++) {
            nfree(net->matrixes[i][j]);
        }
        nfree(net->matrixes[i]);
    }
    nfree(net->matrixes);
}

void showDataMatrix(NeuralNetwork net) {
    for(int32_t i = 0; i < net->hdSize - 1; i++) {
        float **matr = net->matrixes[i];
        for(int32_t j = 0; j < net->structureBuffer[i]; j++) {
            for(int32_t k = 0; k < net->structureBuffer[i + 1]; k++) {
                printf("%f ", matr[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void nn_ShowWeights(NeuralNetwork net) {
    assert(net->numberOfHiddens > 0);
    // for(int32_t i = 0; i < net->hiddensSizes[0]; i++) {
    //     for(int32_t j = 0; j < net->hiddensSizes[1]; j++) {
    //         printf("%f ", getWeight(net->hash, net->inputs[i]->ID, net->hiddens[0][j]->ID));
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for(int32_t i = 0; i < net->numberOfHiddens - 1; i++) {
    //     for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
    //         for(int32_t k = 0; k < net->hiddensSizes[i + 2]; k++) {
    //             printf("%f ", getWeight(net->hash, net->hiddens[i][j]->ID, net->hiddens[i + 1][k]->ID));
    //         }
    //         printf("\n");
    //     }
    // }
}

void nn_ClearNeurons(NeuralNetwork net) {
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            net->hiddens[i][j]->value = 0;
            net->hiddens[i][j]->error = 0;
            net->hiddens[i][j]->unChangedValue = 0;
        }
    }
}

void nn_Mutate(NeuralNetwork self, float chance, float by) {
    int32_t numberOfMutations = (int32_t)(chance * (float)self->maxNeurons) + 1.0;
    assert(self->maxNeurons != 0);
    for(int32_t i = 0; i < numberOfMutations; i++) {
        int32_t fNode = (int32_t)func_RandomNumber(0, (float)(self->maxNeurons) - 0.001);
        Neuron *nodes = NULL;
        int32_t size = 0;
        if(self->allNeurons[fNode]->parents->size > 0) {
            nodes = self->allNeurons[fNode]->parents->buffer;
            size = self->allNeurons[fNode]->parents->size;
        }
        else
        if(self->allNeurons[fNode]->childs->size > 0) {
            nodes = self->allNeurons[fNode]->childs->buffer;
            size = self->allNeurons[fNode]->childs->size;
        }
        int32_t sNode = (int32_t)func_RandomNumber(0, (float)(size) - 0.001);
        float weight = getWeightMatrix(nodes[sNode], fNode, nodes[sNode]->ID);
        float delta = func_RandomNumber(-by, by);
        saveWeightMatrix(nodes[sNode], fNode, nodes[sNode]->ID, weight + delta);
    }
}

float *feedForwardNormal(NeuralNetwork net, float *structureBuffer) {
    int32_t size = net->structureBuffer[0];
    float *result = malloc(sizeof(float) * size);
    nn_ClearNeurons(net);
    for(int32_t i = 0; i < size; i++) {
        net->inputs[i]->value = structureBuffer[i];
        net->inputs[i]->unChangedValue = structureBuffer[i];
        net->inputs[i]->error = 0;
        ne_FeedForward(net->inputs[i]);
    }
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        ne_FeedForward(net->biases[i]);
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_FeedForward(net->hiddens[i][j]);
        }
    }
    for(int32_t i = 0; i < net->hiddensSizes[net->numberOfHiddens]; i++) {
        result[i] = net->hiddens[net->numberOfHiddens - 1][i]->value;
    }
    return result;
}

float *nn_FeedForward(NeuralNetwork net, float *structureBuffer) {
    stat_Switch(net);
    if(!net->train) {
        return feedForwardNormal(net, structureBuffer);
    }
    return feedForwardTrain(net, structureBuffer);
}

float *feedForwardTrain(NeuralNetwork net, float *structureBuffer) {
    int32_t size = net->structureBuffer[0];
    float *result = stat_Alloc(sizeof(float) * size);
    nn_ClearNeurons(net);
    for(int32_t i = 0; i < size; i++) {
        net->inputs[i]->value = structureBuffer[i];
        net->inputs[i]->unChangedValue = structureBuffer[i];
        net->inputs[i]->error = 0;
        ne_FeedForward(net->inputs[i]);
    }
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        ne_FeedForward(net->biases[i]);
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_FeedForward(net->hiddens[i][j]);
        }
    }
    for(int32_t i = 0; i < net->hiddensSizes[net->numberOfHiddens]; i++) {
        result[i] = net->hiddens[net->numberOfHiddens - 1][i]->value;
    }
    return result;
}

float *nesterovFeedForward(NeuralNetwork net, float *structureBuffer) {
    int32_t size = net->structureBuffer[0];
    float *result = malloc(sizeof(float) * size);
    nn_ClearNeurons(net);
    for(int32_t i = 0; i < size; i++) {
        net->inputs[i]->value = structureBuffer[i];
        net->inputs[i]->unChangedValue = structureBuffer[i];
        net->inputs[i]->error = 0;
        ne_NesterovFeedForward(net->inputs[i]);
    }
    for(int32_t i = 0; i < net->numberOfHiddens; i++) {
        ne_NesterovFeedForward(net->biases[i]);
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_NesterovFeedForward(net->hiddens[i][j]);
        }
    }
    for(int32_t i = 0; i < net->hiddensSizes[net->numberOfHiddens]; i++) {
        result[i] = net->hiddens[net->numberOfHiddens - 1][i]->value;
    }
    return result;
}

float nn_Optimize(NeuralNetwork net, float *input, float *output, int8_t type) {
    int32_t outputSize = net->structureBuffer[net->hdSize - 1];
    float *inputResponse = NULL;
    net->train = 1;
    if(type == OPT_SGDNM) {
        inputResponse = nesterovFeedForward(net, input);
    }
    else {
        inputResponse = nn_FeedForward(net, input);
    }
    float totalError = 0;
    for(int32_t i = 0; i < outputSize; i++) {
        float valueError = func_SquaredError(output[i], inputResponse[i]);
        net->hiddens[net->numberOfHiddens - 1][i]->error = valueError;
        totalError += fabs(valueError);
    }
    for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
        for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
            ne_PropagateErrorToParents(net->hiddens[i][j]);
        }
    }
    if(type == OPT_SGD) {
        for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
            for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
                ne_OptimizeSGD(net->hiddens[i][j]);
            }
        }
    }
    if(type == OPT_SGDM || type == OPT_SGDNM) {
        for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
            for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
                ne_OptimizeSgdMomentum(net->hiddens[i][j]);
            }
        }
    }
    if(type == OPT_ADAGRAD) {
        for(int32_t i = net->numberOfHiddens - 1; i >= 0; i--) {
            for(int32_t j = 0; j < net->hiddensSizes[i + 1]; j++) {
                ne_OptimizeAdagrad(net->hiddens[i][j]);
            }
        }
    }
    stat_Free(inputResponse);
    net->train = 0;
    return totalError;
}

void nn_WriteFile(NeuralNetwork net, char *fileName) {
    FILE *fd = fopen(fileName, "w+");
    int32_t sum = func_ArraySum(net->structureBuffer, net->hdSize);
    int32_t *parents = malloc(sum * sum * sizeof(int32_t));
    int32_t *childs = malloc(sum * sum * sizeof(int32_t));
    float *values = malloc(sum * sum * sizeof(float));
    int32_t size = 0;
    for(int32_t i = 0; i < net->maxNeurons; i++) {
        for(int32_t j = 0; j < net->allNeurons[i]->childs->size; j++) {
            Neuron currentChild = ((Neuron *)net->allNeurons[i]->childs->buffer)[j];
            float weight = getWeightMatrix(net->allNeurons[0], i, currentChild->ID);
            if(weight) {
                parents[size] = i;
                childs[size] = currentChild->ID;
                values[size] = weight;
                size++;
            }
        }
    }
    fprintf(fd, "%d\n", size);
    for(int32_t i = 0; i < size; i++) {
        fprintf(fd, "%d %d %f\n", parents[i], childs[i], values[i]);
    }
    free(parents);
    free(childs);
    free(values);
    fclose(fd);
}

int32_t nn_GetProportionalRandomIndex(NeuralNetwork net, float *input) {
    float *response = nn_FeedForward(net, input);
    float sum = 0.0f;
    for(int32_t i = 0; i < net->hdSize; i++) {
        sum += response[i];
    }
    for(int32_t i = 0; i < net->hdSize; i++) {
        response[i] /= sum;
    }
    float randomNumber = func_RandomNumber(0.0f, 1.0f);
    int32_t index = 0;
    while(randomNumber > 0.0f) {
        randomNumber -= response[index];
        index++;
    }
    free(response);
    return index - 1;
}

void nn_LoadFile(NeuralNetwork network, char *fileNanme) {
    FILE *fd = fopen(fileNanme, "r+");
    int32_t totalWeights, a, b;
    float c;
    fscanf(fd, "%d", &totalWeights);
    for(int32_t i = 0; i < totalWeights; i++) {
        fscanf(fd, "%d %d %f", &a, &b, &c);
        saveWeightMatrix(network->allNeurons[0], a, b, c);
    }
    fclose(fd);
}

float elementFromBuffer(float *buffer, int32_t index) {
    return buffer[index];
}

void freeHiddens(NeuralNetwork self) {
    for(int32_t i = 0; i < self->numberOfHiddens; i++) {
        nfree(self->hiddens[i]);
    }
}

void nn_Destroy(NeuralNetwork net) {
    block_Destroy(net);
    get_MemoryLeak();
}