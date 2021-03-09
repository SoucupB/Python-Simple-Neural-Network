#pragma once
#include <stdlib.h>

struct Vector_t {
  void *buffer;
  size_t size;
  size_t capacity;
  size_t objSize;
};

typedef struct Vector_t *Vector;

Vector vct_Init(size_t size);
void vct_Push(Vector self, void *buffer);
void vct_Delete(Vector self);