#ifndef __ceptools_cepfastmarchingcpu_h
#define __ceptools_cepfastmarchingcpu_h

/*
 *  fastmarchingcpu.h
 *
 *  Performing fast marching on the DEM data on the CPU.
 */

#include "cepdb.h"
#include "heap.h"
#include "globals.h"

#define TERRAIN_FACTOR 1.0

/*** Fast marching functions ***/
void fastMarching ( int maxIters );
void fastMarchingInit ( unsigned short int startx, unsigned short int startz );
char fastMarchingStep ();
void fastMarchingCleanup();
signed char heapComparefm ( HeapCoordNode * node1, HeapCoordNode * node2 );
void heapUpdatefm ( HeapCoordNode * coordNode, unsigned int heapIndex );
void updateFringe ( Heap * fringe, unsigned short int indexX, unsigned short int indexZ );
void calculateFastMarchingCost ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * neighborData );

/*** Utility functions ***/
void saveWorkData ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * workData );
void getWorkDataAt ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * workData );
double getElevationAt ( unsigned short int indexX, unsigned short int indexZ );
double length ( double a[3] );

/*** Cost functions ***/
double basicCostFuncCoord ( unsigned short int startIndexx, unsigned short int startIndexz, unsigned short int endIndexx, unsigned short int endIndexz );
double basicCostFuncVec ( double vec[3] );
double caloricCostFuncCoord ( unsigned short int startIndexx, unsigned short int startIndexz, unsigned short int endIndexx, unsigned short int endIndexz );
double caloricCostFuncVec ( double vec[3], double terrainFactor );

#endif
