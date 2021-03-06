#include "Functions.h"
#include <stdlib.h>
#include "StaticAllocator.h"
#include <string.h>

float func_Uniform(float left, float right) {
    float augumentedNumber = (float)rand() + 1;
    float randomNumber = sin((float)rand() * (float)rand() / augumentedNumber);
    return left + (right - left) * fabs(randomNumber);
}

float func_Sigmoid(float value) {
    return 1.0 / (1.0 + exp(-value));
}

float func_DSigmoid(float value) {
    return func_Sigmoid(value) * (1.0 - func_Sigmoid(value));
}

float func_Tanh(float value) {
    return tanh(value);
}

float func_DTanh(float value) {
    float functionValue = func_Tanh(value);
    return 1.0 - functionValue * functionValue;
}

int32_t func_ReadInt32FromFile(FILE *fd) {
    int32_t value;
    char pointer[sizeof(int32_t)];
    fscanf(fd, "%c", &pointer[0]);
    fscanf(fd, "%c", &pointer[1]);
    fscanf(fd, "%c", &pointer[2]);
    fscanf(fd, "%c", &pointer[3]);
    memcpy(&value, pointer, sizeof(int32_t));
    return value;
}

float func_ReadFloatFromFile(FILE *fd) {
    float value;
    char pointer[sizeof(float)];
    fscanf(fd, "%c", &pointer[0]);
    fscanf(fd, "%c", &pointer[1]);
    fscanf(fd, "%c", &pointer[2]);
    fscanf(fd, "%c", &pointer[3]);
    memcpy(&value, pointer, sizeof(float));
    return value;
}

void func_WriteInt32ToFile(FILE *fd, int32_t value) {
    char *buffer = (char *)&value;
    for(int32_t i = 0; i < sizeof(int32_t); i++) {
        fprintf(fd, "%c", buffer[i]);
    }
}

void func_WriteFloatToFile(FILE *fd, float value) {
    char *buffer = (char *)&value;
    for(int32_t i = 0; i < sizeof(int32_t); i++) {
        fprintf(fd, "%c", buffer[i]);
    }
}

float func_Identity(float value) {
    return value;
}

float func_DIdentity(float value) {
    return 1;
}

float func_Relu(float value) {
    return value <= 0.0 ? 0.0 : value;
}

float func_DRelu(float value) {
    return value <= 0.0 ? 0.0 : 1.0;
}

float func_SoftPlus(float value) {
    return log(1.0 + exp(value));
}

float func_DSoftPlus(float value) {
    return 1.0 / (1.0 + exp(-value));
}

float func_ArcTan(float value) {
    return atan(value);
}

float func_DArcTan(float value) {
    return 1.0 / (1.0 + value * value);
}

float func_Gaussian(float value) {
    return exp(-(value * value));
}

float func_DGaussian(float value) {
    return -2 * value * exp(-(value * value));
}

float func_RandomNumber( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

void func_UseSrand() {
    srand(time(NULL));
}

int32_t func_ArraySum(int32_t *buffer, int32_t size) {
    int32_t sum = 0;
    for(int32_t i = 0; i < size; i++) {
        sum += buffer[i];
    }
    return sum;
}

int32_t func_SelectFromProbabilities(float *buffer, int32_t size) { // this array should have values between [0, 1] and their sum should be equal to 1
    int32_t index = 0;
    float randomNumber = func_RandomNumber(0.0, 1.0);
    while(index < size && randomNumber >= 0.0) {
        randomNumber -= buffer[index++];
    }
    assert(index <= size);
    return index - 1;
}

float *func_NormalizeArray(float *buffer, int32_t size) {
    float sumTotal = 0;
    float *normalizedArray = nmalloc(sizeof(float) * size);
    for(int32_t i = 0; i < size; i++) {
        sumTotal += buffer[i];
    }
    for(int32_t i = 0; i < size; i++) {
        normalizedArray[i] = buffer[i] / sumTotal;
    }
    return normalizedArray;
}

float func_SquaredError(float a, float b) {
    return (a < b ? (a - b) * (a - b) * -1 : (a - b) * (a - b));
}

int32_t smin(int32_t a, int32_t b) {
    return (a < b ? a : b);
}

float func_CrossEntropy(float a, float b) {
    return -(a * log(b + 1e-5) + (1 - a) * log(1 - b + 1e-5));
}

int32_t func_TotalFunctions() {
    return 7;
}

void func_freePointer(void *buffer) {
    nfree(buffer);
}

long func_Time() {
    struct timespec _t;
    clock_gettime(CLOCK_REALTIME, &_t);
    return _t.tv_sec * 1000 + lround(_t.tv_nsec / 1.0e6);
}

struct Function_t *func_GetActivationFunctions() {
    struct Function_t *functions = nmalloc(sizeof(struct Function_t) * func_TotalFunctions());
    functions[0].func = func_Sigmoid;
    functions[1].func = func_Tanh;
    functions[2].func = func_Relu;
    functions[3].func = func_Identity;
    functions[4].func = func_SoftPlus;
    functions[5].func = func_ArcTan;
    functions[6].func = func_Gaussian;
    return functions;
}
struct Function_t *func_GetDActivationFunctions() {
    struct Function_t *functions = nmalloc(sizeof(struct Function_t) * func_TotalFunctions());
    functions[0].func = func_DSigmoid;
    functions[1].func = func_DTanh;
    functions[2].func = func_DRelu;
    functions[3].func = func_DIdentity;
    functions[4].func = func_DSoftPlus;
    functions[5].func = func_DArcTan;
    functions[6].func = func_DGaussian;
    return functions;
}