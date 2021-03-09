#include "Vector.h"
#include <string.h>
#include <stdio.h>
#include "StaticAllocator.h"

Vector vct_Init(size_t size) {
  Vector self = nmalloc(sizeof(struct Vector_t));
  self->buffer = malloc(size);
  self->size = 0;
  self->capacity = 1;
  self->objSize = size;
  return self;
}

void copyData(Vector self, void *buffer) {
  memcpy(self->buffer + (self->size * self->objSize), buffer, self->objSize);
  self->size++;
}

void vct_Push(Vector self, void *buffer) {
  if(self->size >= self->capacity) {
    self->capacity <<= 1;
    self->buffer = nrealloc(self->buffer, self->capacity * self->objSize);
  }
  copyData(self, buffer);
}

void vct_Delete(Vector self) {
  nfree(self->buffer);
  nfree(self);
}