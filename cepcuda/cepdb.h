#ifndef __ceptools_cepdb_h
#define __ceptools_cepdb_h

/*
 *  cepdb.h
 *
 *  Structures used to track the data used by the algorithms. 
 */

#define PROP_FROM_LEFT ((unsigned char) 1)
#define PROP_FROM_RIGHT ((unsigned char) 2)
#define PROP_FROM_TOP ((unsigned char) 4)
#define PROP_FROM_BOTTOM ((unsigned char) 8)
#define PROP_TO_LEFT ((unsigned char) 16)
#define PROP_TO_RIGHT ((unsigned char) 32)
#define PROP_TO_TOP ((unsigned char) 64)
#define PROP_TO_BOTTOM ((unsigned char) 128)
#define PROP_FROM_MASK ((unsigned char) 15)
#define PROP_TO_MASK ((unsigned char) 240)

typedef struct DEMHeader_struct
{
    float cellSize;
    unsigned short int width;
    unsigned short int depth;
    float maxElevation;
    float minElevation;
	float xLowerLeftCorner;
	float zLowerLeftCorner;
	float noDataValue;
} DEMHeader;

typedef struct DEMElevationData_struct
{
    float * elevation;
} DEMElevationData;

typedef struct DEMColorData_struct
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
} DEMColorData;

typedef struct DEMWorkData_struct
{
    char visited;
    double calories;
    unsigned int heapIndex;
    unsigned char propagation;
} DEMWorkData;

typedef struct DEMCoords_struct
{
    unsigned short int x;
    unsigned short int z;
} DEMCoords;

#endif
