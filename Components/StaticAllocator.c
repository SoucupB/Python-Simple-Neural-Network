#include "StaticAllocator.h"
#include <stdio.h>
#include "Vector.h"
#include <string.h>

void *segmentMemory;
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
  void *block = malloc(size + sizeof(void *) * 2);
  MemBuffer memStat = mem_Init(sizeof(void **));
  memcpy(block, &memStat, sizeof(MemBuffer));
  return (char *)block + sizeof(void *) * 2;
}

void block_SwitchStatus(void *pointer) {
  void *initOffset = (char *)pointer - sizeof(void *) * 2;
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

void stat_Init(void *pointer, size_t totalMemory) {
  void *stats = malloc(totalMemory);
  void *staticMemory = (char *)pointer - sizeof(void *);
  memcpy(staticMemory, &stats, sizeof(void *));
  segmentMemory = malloc(totalMemory);
}

void stat_Switch(void *pointer) {
  void *staticMemory = (char *)pointer - sizeof(void *);
  memcpy(segmentMemory, staticMemory, sizeof(void *));
}

void *stat_Alloc(size_t totalMemory) {
  allocationsCount++;
  return segmentMemory;
}

void stat_Free(void *buffer) {
  allocationsCount--;
}

void *nrealloc(void *buffer, size_t size) {
  void *newBuffer = nmalloc(sizeof(void *) * size);
  memcpy(newBuffer, buffer, size);
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
  void *initOffset = (char *)pointer - sizeof(void *) * 2;
  void *statMemory = (char *)pointer - sizeof(void *);
  MemBuffer memState = *(MemBuffer *)initOffset;
  void **buffer = memState->buffer;
  for(int32_t i = 0; i < memState->size; i++) {
    nfree(buffer[i]);
  }
  mem_Delete(memState);
  memcpy(segmentMemory, statMemory, sizeof(void *));
  free(segmentMemory);
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