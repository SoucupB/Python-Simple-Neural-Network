#include "StaticAllocator.h"
#include <stdio.h>
#include "Vector.h"
#include <string.h>

int32_t allocationsCount;

struct MemBuffer_t {
  void *buffer;
  size_t size;
  size_t capacity;
  size_t objSize;
};
typedef struct MemBuffer_t *MemBuffer;
MemBuffer usedMemory;

MemBuffer mem_Init(size_t size) {
  MemBuffer self = malloc(sizeof(struct Vector_t));
  self->buffer = malloc(size);
  self->size = 0;
  self->capacity = 1;
  self->objSize = size;
  return self;
}

void *block_Alloc(size_t size) {
  void *block = malloc(size + sizeof(void *));
  MemBuffer memStat = mem_Init(sizeof(void **));
  memcpy(block, &memStat, sizeof(MemBuffer));
  return (char *)block + sizeof(void *);
}

void block_SwitchStatus(void *pointer) {
  void *initOffset = (char *)pointer - sizeof(void *);
  MemBuffer buffer = *(MemBuffer *)initOffset;
  usedMemory = buffer;
}

void copyMemData(MemBuffer self, void *buffer) {
  memcpy(self->buffer + (self->size * self->objSize), buffer, self->objSize);
  self->size++;
}

void mem_Push(MemBuffer self, void *buffer) {
  if(self->size >= self->capacity) {
    self->capacity <<= 1;
    self->buffer = realloc(self->buffer, self->capacity * self->objSize);
  }
  copyMemData(self, buffer);
}

void mem_Delete(MemBuffer self) {
  free(self->buffer);
  free(self);
}

void *nrealloc(void *buffer, size_t previousSize, size_t size) {
  void *newBuffer = nmalloc(size);
  memcpy(newBuffer, buffer, previousSize);
  return newBuffer;
}

void *nmalloc(size_t size) {
  allocationsCount++;
  void *memory = malloc(size);
  mem_Push(usedMemory, &memory);
  return memory;
}

void nfree(void *buffer) {
  allocationsCount--;
  free(buffer);
}

void block_Destroy(void *pointer) {
  void *initOffset = (char *)pointer - sizeof(void *);
  MemBuffer memState = *(MemBuffer *)initOffset;
  void **buffer = memState->buffer;
  for(int32_t i = 0; i < memState->size; i++) {
    nfree(buffer[i]);
  }
  mem_Delete(memState);
  free(initOffset);
}

void get_MemoryLeak() {
  if(!allocationsCount) {
    printf("No memory leaks detected!\n");
  }
  else {
    printf("There are %d memory leaks detected!\n", allocationsCount);
  }
}