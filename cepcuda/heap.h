#ifndef __ceptools_heap_h
#define __ceptools_heap_h

/*
 *  heap.h
 *
 *  Binary heap (priority queue) utility.  Loosly adapted from:
 *  Weiss, Mark Allen. "Data Structures & Algorithm Analysis in Java." Addison Wesley Longman, 1999.
 */

#define HEAP_ARRAY_CHUNK_SIZE 512
#define CHUNK_SHIFT_DIV 9

typedef struct HeapCoordNode_struct
{
    unsigned short int x;
    unsigned short int z;
} HeapCoordNode;

typedef signed char (*heap_compare_func) ( HeapCoordNode *, HeapCoordNode * );
typedef void (*heap_idx_change_func) ( HeapCoordNode *, unsigned int );

typedef struct Heap_struct
{
    unsigned int size;
    heap_compare_func compare;
    heap_idx_change_func update;
    unsigned int numArrays;
    HeapCoordNode ** arrays;
} Heap;

Heap * makeHeap ( heap_compare_func compare, heap_idx_change_func change );
void destroyHeap ( Heap * heap );
void insert ( Heap * heap, unsigned short int indexX, unsigned short int indexZ );
char removeMin ( Heap * heap, unsigned short int * indexX, unsigned short int * indexZ );
void percolateDown ( Heap * heap, unsigned int idx );
void percolateUp ( Heap * heap, unsigned int idx );
void updateElement ( Heap * heap, unsigned int heapIndex );
char getCoordsAt ( Heap * heap, unsigned int heapIndex, unsigned short int * indexX, unsigned short int * indexZ );

#endif
