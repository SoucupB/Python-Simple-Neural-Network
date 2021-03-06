#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Function_t {
    float (*func)(float);
};

void func_UseSrand();
float func_Sigmoid(float value);
float func_DSigmoid(float value);
float func_Uniform(float left, float right);
float func_SquaredError(float a, float b);
float func_CrossEntropy(float a, float b);
struct Function_t *func_GetActivationFunctions();
struct Function_t *func_GetDActivationFunctions();
int32_t func_TotalFunctions();
long func_Time();
void func_FreePointer(void *buffer);
int32_t func_SelectFromProbabilities(float *buffer, int32_t size);
float *func_NormalizeArray(float *buffer, int32_t size);
int32_t func_ArraySum(int32_t *buffer, int32_t size);
float func_RandomNumber(float min, float max);
int32_t smin(int32_t a, int32_t b);
void func_WriteInt32ToFile(FILE *fd, int32_t value);
void func_WriteFloatToFile(FILE *fd, float value);
int32_t func_ReadInt32FromFile(FILE *fd);
float func_ReadFloatFromFile(FILE *fd);

#ifdef __cplusplus
}
#endif