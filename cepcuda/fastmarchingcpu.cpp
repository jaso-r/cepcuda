#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cepdb.h"
#include "fastmarchingcpu.h"


/* Caloric variables, with defaults */
char travelerSex = 1;           /* 0 for female, 1 for male; default male */
double travelerSpeed = 1.25;    /* in meters per second; default 4.5 km/hour */
double travelerAge = 30.0;      /* in years; default 30 years old */
double travelerWeight = 80.0;   /* in kilograms; default about 175 lbs */
double travelerHeight = 183.0;  /* in centimeters; default about 6 ft */
double loadWeight = 0.0;        /* in kilograms; default no load */
double greatestCalories = 0.0;

DEMHeader demHeader;
DEMElevationData demData;
DEMWorkData* workDatas;
Heap * fringe = 0;

// initializaiton for running fast marching on the cpu
void fastMarchingInit(unsigned short int startx, unsigned short int startz)
{
	fringe = makeHeap(heapComparefm, heapUpdatefm);
	DEMWorkData workData;

	printf("Starting at %u, %u\n", startx, startz);

	getWorkDataAt(startx, startz, &workData);
	workData.calories = 0.0;
	workData.visited = 1;
	workData.heapIndex = 0;
	workData.propagation = 0;
	saveWorkData(startx, startz, &workData);

	updateFringe(fringe, startx, startz);
}

// "steps" fast maching
char fastMarchingStep()
{
	DEMWorkData workData;
	unsigned short int iterX;
	unsigned short int iterZ;
	char ret = removeMin(fringe, &iterX, &iterZ);

	if (ret)
	{
		getWorkDataAt(iterX, iterZ, &workData);
		workData.visited = 1;
		workData.heapIndex = 0;
		if (workData.calories > greatestCalories)
		{
			greatestCalories = workData.calories;
		}
		saveWorkData(iterX, iterZ, &workData);

		updateFringe(fringe, iterX, iterZ);

		return 0;
	}
	else
	{
		fastMarchingCleanup();
		printf("\nDone with fast marching.\n");
		return 1;
	}
}

// cleanup data used by fast marching
void fastMarchingCleanup()
{
	if (fringe)
	{
		destroyHeap(fringe);
		fringe = 0;
	}
}

// runs the entire fast marching algorithm
void fastMarching ( int maxIters )
{
    unsigned int nodeCount = 1;
    DEMWorkData workData;
    char ret;
    unsigned short int iterX;
    unsigned short int iterZ;
    
    printf("Running fast marching algorithm on the CPU to compute caloric costs.\n");
    
    ret = removeMin(fringe, &iterX, &iterZ);

    while (ret && (maxIters < 0 || nodeCount < maxIters))
    {
        getWorkDataAt(iterX, iterZ, &workData);
        workData.visited = 1;
        workData.heapIndex = 0;
        if (workData.calories > greatestCalories)
        {
            greatestCalories = workData.calories;
        }
        saveWorkData(iterX, iterZ, &workData);
                
        updateFringe(fringe, iterX, iterZ);
        nodeCount++;
        
        ret = removeMin(fringe, &iterX, &iterZ);
        
        if (nodeCount % demHeader.width == 0)
        {
            printf(".");
            fflush(stdout);
        }
    }
    
	fastMarchingCleanup();
    
    printf("\nDone with fast marching after %u iterations.\n", nodeCount);
}

signed char heapComparefm ( HeapCoordNode * node1, HeapCoordNode * node2 )
{
    DEMWorkData data1;
    DEMWorkData data2;
    signed char returnVal = 0;
    
    getWorkDataAt(node1->x, node1->z, &data1);
    getWorkDataAt(node2->x, node2->z, &data2);
    
    if (data1.calories < data2.calories)
    {
        returnVal = -1;
    }
    else if (data1.calories > data2.calories)
    {
        returnVal = 1;
    }
        
    return returnVal;
}

void heapUpdatefm ( HeapCoordNode * coordNode, unsigned int heapIndex )
{
    DEMWorkData workData;
    getWorkDataAt(coordNode->x, coordNode->z, &workData);
    workData.heapIndex = heapIndex + 1;
    saveWorkData(coordNode->x, coordNode->z, &workData);
}

void updateFringe ( Heap * fringe, unsigned short int indexX, unsigned short int indexZ )
{
    DEMWorkData neighborData;
    int i, j, iter;
    unsigned short int sanityX;
    unsigned short int sanityZ;
    
    for (iter = 0; iter < 4; iter++)
    {
        switch (iter)
        {
            case 0:
                i = -1;
                j = 0;
                break;
            case 1:
                i = 1;
                break;
            case 2:
                i = 0;
                j = -1;
                break;
            case 3:
                j = 1;
                break;
        }
        
        if (((int) indexX) + i >= 0 && ((int) indexX) + i < ((int) demHeader.width) && ((int) indexZ) + j >= 0 && ((int) indexZ) + j < ((int) demHeader.depth))
        {
            getWorkDataAt((unsigned short int) (((int) indexX) + i), (unsigned short int) (((int) indexZ) + j), &neighborData);
            if (!(neighborData.visited))
            {
                calculateFastMarchingCost((unsigned short int) (((int) indexX) + i), (unsigned short int) (((int) indexZ) + j), &neighborData);
                if (neighborData.heapIndex != 0)
                {
                    if (!(getCoordsAt(fringe, neighborData.heapIndex - 1, &sanityX, &sanityZ)))
                    {
                        printf("Bad heap index.\n");
                        exit(-1);
                    }
                    if (sanityX != (unsigned short int) (((int) indexX) + i) || sanityZ != (unsigned short int) (((int) indexZ) + j))
                    {
                        printf("Incorrect heap index for the coords. Wanted (%d, %d). Found (%d, %d).\n", (((int) indexX) + i), (((int) indexZ) + j), (int) sanityX, (int) sanityZ);
                        exit(-1);
                    }
                    updateElement(fringe, neighborData.heapIndex - 1);
                }
                else
                {
                    insert(fringe, (unsigned short int) (((int) indexX) + i), (unsigned short int) (((int) indexZ) + j));
                }
            }
        }
    }
}

void calculateFastMarchingCost ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * neighborData )
{
    DEMWorkData * neighborXData = NULL;
    DEMWorkData * neighborZData = NULL;
    DEMWorkData neighborData1;
    DEMWorkData neighborData2;
    DEMWorkData neighborData3;
    DEMWorkData neighborData4;
    double costX = 0.0;
    double costZ = 0.0;
    double tmpCost = 0.0;
    double vec[3];
    double a;
    double b;
    double c;
    double solution1;
    double solution2;
    double finalCost;
    double rad;
    double approxSol = -1.0;
    double xSol = -1.0;
    double zSol = -1.0;
    
    neighborData1.calories = -1.0;
    neighborData2.calories = -1.0;
    neighborData3.calories = -1.0;
    neighborData4.calories = -1.0;
    
    neighborData->propagation = neighborData->propagation & PROP_TO_MASK;
    
    if (((int) indexX) - 1 >= 0)
    {
        getWorkDataAt(indexX - 1, indexZ, &neighborData1);
        if (neighborData1.calories != -1.0)
        {
            vec[0] = demHeader.cellSize;
            vec[1] = getElevationAt(indexX, indexZ) - getElevationAt(indexX - 1, indexZ);
            vec[2] = 0.0;
            costX = caloricCostFuncVec(vec, TERRAIN_FACTOR);
            neighborXData = &neighborData1;
            neighborData->propagation = neighborData->propagation | PROP_FROM_LEFT;
        }
    }
    
    if (((int) indexX) + 1 < demHeader.width)
    {
        getWorkDataAt(indexX + 1, indexZ, &neighborData2);
        if (neighborData2.calories != -1.0)
        {
            vec[0] = -1.0 * demHeader.cellSize;
            vec[1] = getElevationAt(indexX, indexZ) - getElevationAt(indexX + 1, indexZ);
            vec[2] = 0.0;
            tmpCost = caloricCostFuncVec(vec, TERRAIN_FACTOR);
            if (neighborXData == NULL || neighborData2.calories + tmpCost < neighborXData->calories + costX)
            {
                if (neighborData->propagation & PROP_FROM_LEFT)
                {
                    neighborData->propagation = neighborData->propagation & PROP_TO_MASK;
                }
                neighborData->propagation = neighborData->propagation | PROP_FROM_RIGHT;
                costX = tmpCost;
                neighborXData = &neighborData2;
            }
        }
    }
    
    if (((int) indexZ) - 1 >= 0)
    {
        getWorkDataAt(indexX, indexZ - 1, &neighborData3);
        if (neighborData3.calories != -1.0)
        {
            vec[0] = 0.0;
            vec[1] = getElevationAt(indexX, indexZ) - getElevationAt(indexX, indexZ - 1);
            vec[2] = demHeader.cellSize;
            costZ = caloricCostFuncVec(vec, TERRAIN_FACTOR);
            neighborZData = &neighborData3;
            neighborData->propagation = neighborData->propagation | PROP_FROM_TOP;
        }
    }
    
    if (((int) indexZ) + 1 < demHeader.depth)
    {
        getWorkDataAt(indexX, indexZ + 1, &neighborData4);
        if (neighborData4.calories != -1.0)
        {
            vec[0] = 0.0;
            vec[1] = getElevationAt(indexX, indexZ) - getElevationAt(indexX, indexZ + 1);
            vec[2] = -1.0 * demHeader.cellSize;
            tmpCost = caloricCostFuncVec(vec, TERRAIN_FACTOR);
            if (neighborZData == NULL || neighborData4.calories + tmpCost < neighborZData->calories + costZ)
            {
                if (neighborData->propagation & PROP_FROM_TOP)
                {
                    neighborData->propagation = neighborData->propagation & (PROP_TO_MASK | PROP_FROM_RIGHT | PROP_FROM_LEFT);
                }
                neighborData->propagation = neighborData->propagation | PROP_FROM_BOTTOM;
                costZ = tmpCost;
                neighborZData = &neighborData4;
            }
        }
    }
    
    if (neighborXData != NULL && neighborZData != NULL)
    {
        a = (1.0 / (costX * costX)) + (1.0 / (costZ * costZ));
        b = ((-2.0 * neighborXData->calories) / (costX * costX)) + ((-2.0 * neighborZData->calories) / (costZ * costZ));
        c = ((neighborXData->calories * neighborXData->calories) / (costX * costX)) + ((neighborZData->calories * neighborZData->calories) / (costZ * costZ)) - 1.0;
        
        rad = (b * b) + (-4.0 * a * c);
        
        if (rad < 0.0)
        {
            approxSol = (-1.0 * b) / (2.0 * a);
        }
        else
        {
            solution1 = ((-1.0 * b) + sqrt(rad)) / (2.0 * a);
            solution2 = ((-1.0 * b) - sqrt(rad)) / (2.0 * a);
            
            if (solution1 > solution2)
            {
                approxSol = solution1;
            }
            else
            {
                approxSol = solution2;
            }
        }
    }
    else if (neighborZData != NULL && neighborXData == NULL)
    {
        a = (1.0 / (costZ * costZ));
        b = ((-2.0 * neighborZData->calories) / (costZ * costZ));
        c = ((neighborZData->calories * neighborZData->calories) / (costZ * costZ)) - 1.0;
        
        rad = (b * b) + (-4.0 * a * c);
        
        if (rad < 0.0)
        {
            zSol = (-1.0 * b) / (2.0 * a);
        }
        else
        {
            solution1 = ((-1.0 * b) + sqrt(rad)) / (2.0 * a);
            solution2 = ((-1.0 * b) - sqrt(rad)) / (2.0 * a);
            
            if (solution1 > solution2)
            {
                zSol = solution1;
            }
            else
            {
                zSol = solution2;
            }
        }
    }
    else if (neighborXData != NULL && neighborZData == NULL)
    {
        a = (1.0 / (costX * costX));
        b = ((-2.0 * neighborXData->calories) / (costX * costX));
        c = ((neighborXData->calories * neighborXData->calories) / (costX * costX)) - 1.0;
        
        rad = (b * b) + (-4.0 * a * c);
        
        if (rad < 0.0)
        {
            xSol = (-1.0 * b) / (2.0 * a);
        }
        else
        {
            solution1 = ((-1.0 * b) + sqrt(rad)) / (2.0 * a);
            solution2 = ((-1.0 * b) - sqrt(rad)) / (2.0 * a);
            
            if (solution1 > solution2)
            {
                xSol = solution1;
            }
            else
            {
                xSol = solution2;
            }
        }
    }
    else
    {
        printf("Badness.\n");
        exit(-1);
    }
    
    finalCost = approxSol;
    if (finalCost == -1.0 || (zSol < finalCost && zSol != -1.0))
    {
        finalCost = zSol;
    }
    if (finalCost == -1.0 || (xSol < finalCost && xSol != -1.0))
    {
        finalCost = xSol;
    }
    if (finalCost == -1.0)
    {
        printf("This shouldn't happen!\n");
        exit(-1);
    }
    
    neighborData->calories = finalCost;
    saveWorkData(indexX, indexZ, neighborData);
    
    if (neighborData1.calories != -1.0)
    {
        if (neighborData->propagation & PROP_FROM_LEFT && !(neighborData1.propagation & PROP_TO_RIGHT))
        {
            neighborData1.propagation = neighborData1.propagation | PROP_TO_RIGHT;
            saveWorkData(indexX - 1, indexZ, &neighborData1);
        }
        else if (!(neighborData->propagation & PROP_FROM_LEFT) && neighborData1.propagation & PROP_TO_RIGHT)
        {
            neighborData1.propagation = neighborData1.propagation & (PROP_FROM_MASK | PROP_TO_LEFT | PROP_TO_TOP | PROP_TO_BOTTOM);
            saveWorkData(indexX - 1, indexZ, &neighborData1);
        }
    }
    
    if (neighborData2.calories != -1.0)
    {
        if (neighborData->propagation & PROP_FROM_RIGHT && !(neighborData2.propagation & PROP_TO_LEFT))
        {
            neighborData2.propagation = neighborData2.propagation | PROP_TO_LEFT;
            saveWorkData(indexX + 1, indexZ, &neighborData2);
        }
        else if (!(neighborData->propagation & PROP_FROM_RIGHT) && neighborData2.propagation & PROP_TO_LEFT)
        {
            neighborData2.propagation = neighborData2.propagation & (PROP_FROM_MASK | PROP_TO_RIGHT | PROP_TO_TOP | PROP_TO_BOTTOM);
            saveWorkData(indexX + 1, indexZ, &neighborData2);
        }
    }
    
    if (neighborData3.calories != -1.0)
    {
        if (neighborData->propagation & PROP_FROM_TOP && !(neighborData3.propagation & PROP_TO_BOTTOM))
        {
            neighborData3.propagation = neighborData3.propagation | PROP_TO_BOTTOM;
            saveWorkData(indexX, indexZ - 1, &neighborData3);
        }
        else if (!(neighborData->propagation & PROP_FROM_TOP) && neighborData3.propagation & PROP_TO_BOTTOM)
        {
            neighborData3.propagation = neighborData3.propagation & (PROP_FROM_MASK | PROP_TO_RIGHT | PROP_TO_TOP | PROP_TO_LEFT);
            saveWorkData(indexX, indexZ - 1, &neighborData3);
        }
    }
    
    if (neighborData4.calories != -1.0)
    {
        if (neighborData->propagation & PROP_FROM_BOTTOM && !(neighborData4.propagation & PROP_TO_TOP))
        {
            neighborData4.propagation = neighborData4.propagation | PROP_TO_TOP;
            saveWorkData(indexX, indexZ + 1, &neighborData4);
        }
        else if (!(neighborData->propagation & PROP_FROM_BOTTOM) && neighborData4.propagation & PROP_TO_TOP)
        {
            neighborData4.propagation = neighborData4.propagation & (PROP_FROM_MASK | PROP_TO_RIGHT | PROP_TO_BOTTOM | PROP_TO_LEFT);
            saveWorkData(indexX, indexZ + 1, &neighborData4);
        }
    }
}

void saveWorkData ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * workData )
{
	memcpy(&(workDatas[(indexZ * demHeader.width) + indexX]), workData, sizeof(DEMWorkData));
}

void getWorkDataAt ( unsigned short int indexX, unsigned short int indexZ, DEMWorkData * workData )
{
	memcpy(workData, &(workDatas[(indexZ * demHeader.width) + indexX]), sizeof(DEMWorkData));
}

double getElevationAt ( unsigned short int indexX, unsigned short int indexZ )
{
	return demData.elevation[(indexZ * demHeader.width) + indexX];
}

double length ( double a[3] )
{
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

double basicCostFuncCoord ( unsigned short int startIndexx, unsigned short int startIndexz, unsigned short int endIndexx, unsigned short int endIndexz )
{
    double vec[3];
    vec[0] = ((double)(endIndexx - startIndexx)) * demHeader.cellSize;
    vec[1] = getElevationAt(endIndexx, endIndexz) - getElevationAt(startIndexx, startIndexz);
    vec[2] = ((double)(endIndexz - startIndexz)) * demHeader.cellSize;
    return basicCostFuncVec(vec);
}

double basicCostFuncVec ( double vec[3] )
{
    return length(vec) / demHeader.cellSize;
}

double caloricCostFuncCoord ( unsigned short int startIndexx, unsigned short int startIndexz, unsigned short int endIndexx, unsigned short int endIndexz )
{
    double vec[3];
    vec[0] = ((double)(endIndexx - startIndexx)) * demHeader.cellSize;
    vec[1] = getElevationAt(endIndexx, endIndexz) - getElevationAt(startIndexx, startIndexz);
    vec[2] = ((double)(endIndexz - startIndexz)) * demHeader.cellSize;
    return caloricCostFuncVec(vec, TERRAIN_FACTOR);
}

double caloricCostFuncVec ( double vec[3], double terrainFactor )
{
	double euclidDistance;
	double rise;
	double run;
	double percentGrade;
	double LOverWSquared;
	double VSquared;
	double GPlus6Squared;
	double WPlusL;
	double wattsPE;
	double wattsCF;
	double watts;
	double minutesRequired;
	double caloriesPerMinute;
	double kgcalories;
	double basalKcalPerMinute;
	double basalKcal;
    
    euclidDistance = length(vec);
    rise = vec[1];
    vec[1] = 0.0;
    run = length(vec);
    percentGrade = (rise / run) * 100.0;
    
    LOverWSquared = pow((loadWeight / travelerWeight), 2.0);
    VSquared = travelerSpeed * travelerSpeed;
    GPlus6Squared = pow((percentGrade + 6.0), 2.0);
    WPlusL = travelerWeight + loadWeight;
    wattsPE = 1.5 * travelerWeight + 2.0 * WPlusL * LOverWSquared
        + terrainFactor * WPlusL * (1.5 * VSquared + 0.35 * travelerSpeed * percentGrade);
    wattsCF = terrainFactor * ((percentGrade * WPlusL * travelerSpeed) / 3.5 
                               - (WPlusL * GPlus6Squared / travelerWeight) + (25.0 - VSquared));
    
    if (percentGrade >= 0.0)
    {
        watts = wattsPE;
    }
    else
    {
        watts = wattsPE - wattsCF;
    }
    
    minutesRequired = euclidDistance / (travelerSpeed * 60.0);
    caloriesPerMinute = watts * 0.01433;
    kgcalories = caloriesPerMinute * minutesRequired;
    
    /* Harris-Benedict equation for Basal Metabolism (Kcals per day)
     * Males: 66 + (13.7 * WeightKg) + (5 * HeightCM) - (6.8* Age)
     * Females: 655 + (9.6 * WeightKg) + (1.7 * HeightCM) - (4.7 * Age) */
    if (travelerSex)
    {
        basalKcalPerMinute = (66.0 + (13.7 * travelerWeight) + (5.0 * travelerHeight) - (6.8 * travelerAge)) / 1440.0;
    }
    else
    {
        basalKcalPerMinute = (655.0 + (9.6 * travelerWeight) + (1.7 * travelerHeight) - (4.7 * travelerAge)) / 1440.0;
    }
    basalKcal = basalKcalPerMinute * minutesRequired;
    
    if (kgcalories < basalKcal)
    {
        kgcalories = basalKcal;
    }
    
    return kgcalories;
}
