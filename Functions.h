#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>

struct Function_t {
    float (*func)(float);
};

float func_Sigmoid(float value);
float func_DSigmoid(float value);
float func_Uniform(float left, float right);
float func_SquaredError(float a, float b);
float func_CrossEntropy(float a, float b);
struct Function_t *func_GetActivationFunctions();
struct Function_t *func_GetDActivationFunctions();
int32_t func_TotalFunctions();
float func_Gaussian();