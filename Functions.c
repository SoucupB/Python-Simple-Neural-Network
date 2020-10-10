#include "Functions.h"
#include <stdlib.h>

float func_Uniform(float left, float right) {
    float randomNumber = sin((float)rand() * (float)rand() / (float)rand());
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

float func_Relu(float value) {
    return value <= 0 ? 0 : value;
}

float func_DRelu(float value) {
    return value <= 0 ? 0 : 1;
}

float func_SquaredError(float a, float b) {
    return (a < b ? (a - b) * (a - b) * -1 : (a - b) * (a - b));
}

float func_CrossEntropy(float a, float b) {
    return -(a * log(b) + (1 - a) * log(1 - b + 1e-5));
}

int32_t func_TotalFunctions() {
    return 3;
}

struct Function_t *func_GetActivationFunctions() {
    struct Function_t *functions = malloc(sizeof(struct Function_t) * func_TotalFunctions());
    functions[0].func = func_Sigmoid;
    functions[1].func = func_Tanh;
    functions[2].func = func_Relu;
    return functions;
}
struct Function_t *func_GetDActivationFunctions() {
    struct Function_t *functions = malloc(sizeof(struct Function_t) * func_TotalFunctions());
    functions[0].func = func_DSigmoid;
    functions[1].func = func_DTanh;
    functions[2].func = func_DRelu;
    return functions;
}