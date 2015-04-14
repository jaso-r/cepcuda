#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "makedb.h"
#include "cepdb.h"

void loadDEMData (DEMHeader * headerdb, DEMElevationData * demdb, FILE * demFile)
{
    const char delims[3] = {'\n', ' ', '\0'};
    char lineIn[80];
    int row = 0;
    int column = 0;
    int demLineSize;
    char * demLine;
    char * token;
    float elevation;
    int done = 0;
    int maxReadWidth = 0;

	printf("Loading DEM data...\n");

	headerdb->maxElevation = FLT_MIN;
	headerdb->minElevation = FLT_MAX;
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
    headerdb->width = atoi(strtok(NULL, delims));
	//headerdb->width = 25;
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
	headerdb->depth = atoi(strtok(NULL, delims));
	//headerdb->depth = 25;
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
    headerdb->xLowerLeftCorner = atof(strtok(NULL, delims));
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
	headerdb->zLowerLeftCorner = atof(strtok(NULL, delims));
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
	headerdb->cellSize = atof(strtok(NULL, delims));
    /* only convert if the cell size is actually in degrees (likely less than 1), not meters (likely greater than 1) */
	if (headerdb->cellSize < 1.0)
    {
		headerdb->cellSize = convertCellSizeToMeters(headerdb->cellSize, headerdb->xLowerLeftCorner, headerdb->zLowerLeftCorner);
    }
    
    fgets(lineIn, 80, demFile);
    strtok(lineIn, delims);
	headerdb->noDataValue = atof(strtok(NULL, delims));
    
	demLineSize = headerdb->width * 10;
    demLine = (char *) malloc(sizeof(char) * demLineSize);

	demdb->elevation = (float *) malloc(sizeof(float) * headerdb->depth * headerdb->width);
    
    /* read all the elevation data in from the file */
	for (row = 0; row < headerdb->depth; row++)
	{
		fgets(demLine, demLineSize, demFile);
		token = strtok(demLine, delims);

		for (column = 0; column < headerdb->width; column++)
		{
			elevation = atof(token);

			if (row % 100 == 0 && column == 0)
			{
				printf(".");
				fflush(stdout);
			}

			if (elevation != headerdb->noDataValue)
			{
				if (elevation > headerdb->maxElevation)
				{
					headerdb->maxElevation = elevation;
				}
				if (elevation < headerdb->minElevation)
				{
					headerdb->minElevation = elevation;
				}
			}

			demdb->elevation[(row * headerdb->width) + column] = elevation;

			token = strtok(NULL, delims);
		}
	}
    
    free(demLine);
	demLine = NULL;
    
    printf("\nDone loading DEM data\n");
}

/*
 * All of this function is copied from Brian Wood's getCellSizeInMeters function.
 */
double convertCellSizeToMeters ( double cellSizeInDegrees, double xLowerLeft, double zLowerLeft )
{
    double AlatDD;
    double AlongDD;
    double BlatDD;
    double BlongDD;
    
    double AlatRadians;
    double AlongRadians;
    double BlatRadians;
    double BlongRadians;
    
    double r;
	
    /*
     * using the following formula
     * Distance between point A and point B
     *
     * radians = PI*degree/180
     */
    
    /*
     * Dist(A,B) = r*arcos[ sin(A.lat)*sin(B.lat) + cos(A.lat)*cos(B.lat)*cos(A.long-B.long) ]
     * where
     * r = radius of earth in meters
     * and
     * lattitude and longitude degree values are expressed in radians
     *
     * The logic I use to find the cellSize in meters is to get the distance between two cells that are on the same
     * horizontal profile and only differ by cellSize amount in degrees of lattitude
     */
    
    /*
     * this is from nasa
     * Differences in decimal degrees were converted to meters based on 
     * the WGS-84 Equatorial Radius (link to the WWW) of 6378137.0 meters. 
     * Degrees of latitude, and of longitude at the equator, are thus equal to 111319.5 meters. 
     * Applying a cosine correction as a function of latitude, 
     * degrees of longitude are equal to 
     * 110895.9 m at +/- 5‚àû, 
     * 109628.3 m at +/- 10‚àû, 
     * 107526.4 m at +/- 15‚àû, 
     * and 104606.1 m at +/- 20‚àû. 
     */
    
    AlatDD = xLowerLeft;
    AlongDD = zLowerLeft;
    BlatDD = xLowerLeft + cellSizeInDegrees;
    BlongDD = zLowerLeft;
    
    AlatRadians = 3.14159265 * AlatDD / 180;
    AlongRadians = 3.14159265 * AlongDD / 180;
    BlatRadians = 3.14159265 * BlatDD / 180;
    BlongRadians = 3.14159265 * BlongDD / 180;
    
    r = 6378137; /* earth's radius according to WGS 1984 */
    
    return r * acos(sin(AlatRadians) * sin(BlatRadians) + cos(AlatRadians) * cos(BlatRadians) * cos(AlongRadians - BlongRadians));
}
