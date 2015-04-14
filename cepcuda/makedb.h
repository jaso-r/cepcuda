#ifndef __ceptools_cepmakedb_h
#define __ceptools_cepmakedb_h

#include <stdio.h>
#include "cepdb.h"

/*
 *  makedb.h
 *
 *  Loads in DEM data from a file of a particular text format.
 */

typedef struct DEMFileInfo_struct
{
    char * filename;
    FILE * demFile;
    float xLowerLeftCorner;
    float zLowerLeftCorner;
    float cellSizeMeters;
    float noDataValue;
    int readWidth;
    int readDepth;
} DEMFileInfo;

void loadDEMData ( DEMHeader * headerdb, DEMElevationData * demdb, FILE * demFile );
double convertCellSizeToMeters ( double cellSizeInDegrees, double xLowerLeft, double zLowerLeft );

#endif
