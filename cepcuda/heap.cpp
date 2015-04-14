#include <stdlib.h>
#include <stdio.h>
#include "heap.h"

Heap * makeHeap ( heap_compare_func compare, heap_idx_change_func change )
{
    Heap * heap = (Heap *) malloc(sizeof(struct Heap_struct));
    heap->size = 0;
    heap->compare = compare;
    heap->update = change;
    heap->numArrays = 1;
    heap->arrays = (HeapCoordNode **) malloc(sizeof(HeapCoordNode *));
    heap->arrays[0] = (HeapCoordNode *) malloc(sizeof(struct HeapCoordNode_struct) * HEAP_ARRAY_CHUNK_SIZE);
    return heap;
}

void destroyHeap ( Heap * heap )
{
    int i;
    for (i = 0; i < heap->numArrays; i++)
    {
        free(heap->arrays[i]);
    }
    free(heap);
}

void insert ( Heap * heap, unsigned short int indexX, unsigned short int indexZ )
{
    unsigned int hole;
    HeapCoordNode ** newArrays = NULL;
    int i;
    
    if (heap->size == heap->numArrays * HEAP_ARRAY_CHUNK_SIZE)
    {
        heap->numArrays += 1;
        newArrays = (HeapCoordNode **) malloc(sizeof(HeapCoordNode *) * heap->numArrays);
        if (newArrays == NULL)
        {
            perror("malloc newArrays");
            fprintf(stderr, "Heap size was %d\n", (int) heap->size);
            exit(-1);
        }
        for (i = 0; i < heap->numArrays - 1; i++)
        {
            newArrays[i] = heap->arrays[i];
        }
        newArrays[heap->numArrays - 1] = (HeapCoordNode *) malloc(sizeof(struct HeapCoordNode_struct) * HEAP_ARRAY_CHUNK_SIZE);
        if (newArrays[heap->numArrays - 1] == NULL)
        {
            perror("malloc new array");
            fprintf(stderr, "Heap size was %d\n", (int) heap->size);
            exit(-1);
        }
        free(heap->arrays);
        heap->arrays = newArrays;
    }
    
    hole = heap->size;
    heap->size += 1;
    
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x = indexX;
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z = indexZ;
    heap->update(&(heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE]), hole);
    
    percolateUp(heap, hole);
}

char removeMin ( Heap * heap, unsigned short int * indexX, unsigned short int * indexZ )
{    
    if (heap->size == 0)
    {
        return 0;
    }
    
    *indexX = heap->arrays[0][0].x;
    *indexZ = heap->arrays[0][0].z;
    heap->size -= 1;
    heap->arrays[0][0].x = heap->arrays[heap->size >> CHUNK_SHIFT_DIV][heap->size % HEAP_ARRAY_CHUNK_SIZE].x;
    heap->arrays[0][0].z = heap->arrays[heap->size >> CHUNK_SHIFT_DIV][heap->size % HEAP_ARRAY_CHUNK_SIZE].z;
    
    heap->update(&(heap->arrays[0][0]), 0);
    
    percolateDown(heap, 0);
    
    if (heap->numArrays > 1 && heap->size == (heap->numArrays - 1) * HEAP_ARRAY_CHUNK_SIZE)
    {
        free(heap->arrays[heap->numArrays - 1]);
        heap->numArrays -= 1;
    }
    
    return 1;
}

void percolateUp ( Heap * heap, unsigned int idx )
{
    unsigned int hole = idx;
    HeapCoordNode insertCoord;
    
    insertCoord.x = heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x;
    insertCoord.z = heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z;
    
    for (; hole >= 1 && heap->compare(&insertCoord, &(heap->arrays[(hole >> 1) >> CHUNK_SHIFT_DIV][(hole >> 1) % HEAP_ARRAY_CHUNK_SIZE])) < 0; hole = hole >> 1)
    {
        heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x = heap->arrays[(hole >> 1) >> CHUNK_SHIFT_DIV][(hole >> 1) % HEAP_ARRAY_CHUNK_SIZE].x;
        heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z = heap->arrays[(hole >> 1) >> CHUNK_SHIFT_DIV][(hole >> 1) % HEAP_ARRAY_CHUNK_SIZE].z;
        heap->update(&(heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE]), hole);
    }
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x = insertCoord.x;
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z = insertCoord.z;
    heap->update(&(heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE]), hole);
}

void percolateDown ( Heap * heap, unsigned int idx )
{
    unsigned int hole = idx;
    unsigned int child;
    HeapCoordNode tmp;
    tmp.x = heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x;
    tmp.z = heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z;
    
    for (; hole << 1 < heap->size; hole = child)
    {
        child = hole << 1;
        if (child != heap->size - 1 && heap->compare(&(heap->arrays[(child + 1) >> CHUNK_SHIFT_DIV][(child + 1) % HEAP_ARRAY_CHUNK_SIZE]), &(heap->arrays[child >> CHUNK_SHIFT_DIV][child % HEAP_ARRAY_CHUNK_SIZE])) < 0)
        {
            child += 1;
        }
        if (heap->compare(&(heap->arrays[child >> CHUNK_SHIFT_DIV][child % HEAP_ARRAY_CHUNK_SIZE]), &tmp) < 0)
        {
            heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x = heap->arrays[child >> CHUNK_SHIFT_DIV][child % HEAP_ARRAY_CHUNK_SIZE].x;
            heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z = heap->arrays[child >> CHUNK_SHIFT_DIV][child % HEAP_ARRAY_CHUNK_SIZE].z;
            heap->update(&(heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE]), hole);
        }
        else
        {
            break;
        }
    }
    
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].x = tmp.x;
    heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE].z = tmp.z;
    heap->update(&(heap->arrays[hole >> CHUNK_SHIFT_DIV][hole % HEAP_ARRAY_CHUNK_SIZE]), hole);
}

void updateElement ( Heap * heap, unsigned int heapIndex )
{
    if (heapIndex >= 1 && heap->compare(&(heap->arrays[heapIndex >> CHUNK_SHIFT_DIV][heapIndex % HEAP_ARRAY_CHUNK_SIZE]), &(heap->arrays[(heapIndex >> 1) >> CHUNK_SHIFT_DIV][(heapIndex >> 1) % HEAP_ARRAY_CHUNK_SIZE])) < 0)
    {
        percolateUp(heap, heapIndex);
    }
    else
    {
        percolateDown(heap, heapIndex);
    }
}

char getCoordsAt ( Heap * heap, unsigned int heapIndex, unsigned short int * indexX, unsigned short int * indexZ )
{
    if (heapIndex >= heap->size)
    {
        return 0;
    }
    
    *indexX = heap->arrays[heapIndex >> CHUNK_SHIFT_DIV][heapIndex % HEAP_ARRAY_CHUNK_SIZE].x;
    *indexZ = heap->arrays[heapIndex >> CHUNK_SHIFT_DIV][heapIndex % HEAP_ARRAY_CHUNK_SIZE].z;
    
    return 1;
}
