#pragma once
#include <stdlib.h>
#include <stdint.h>

void *stat_Alloc(size_t size);
void stat_Init(void *pointer, size_t totalMemory);
void stat_Free(void *buffer);
void *nmalloc(size_t size);
void nfree(void *buffer);
void get_MemoryLeak();
void block_Destroy(void *pointer);
void init_Allocations();
void *nrealloc(void *buffer, size_t previousSize, size_t size);
void *block_Alloc(size_t size);
void block_SwitchStatus(void *pointer);
void block_Free(void *pointer);
void stat_Switch(void *pointer);