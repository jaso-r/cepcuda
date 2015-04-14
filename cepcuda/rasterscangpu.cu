#ifndef __ceptools_rasterscangpu_cu
#define __ceptools_rasterscangpu_cu

/*
*  rasterscangpu.cu
*
*  CUDA code used to calculate caloric geodesics using a parallel raster-scan approach.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include "cepdb.h"
#include "globals.h"

#define TILE_DIM     24
#define THREAD_COUNT 512
#define STRIP_HEIGHT 16

float* workCals;
float* demElevs;
texture<float, 2> demTex;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// this was mostly just copied from the CUDA examples.  The idea is to transpose
// the data so that running the raster scan from left to right and from right to left
// would be more performant.  However, this transpose step seems pretty slow itself.
// Maybe there's a way to avoid it or otherwise optomize more.
__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int blockIdx_x, blockIdx_y;

	// do diagonal reordering
	if (width == height)
	{
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else
	{
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;

	int index_in = xIndex + (yIndex * width);
	int index_out = yIndex + (xIndex * height);

	if (xIndex < width && yIndex < height)
	{
		tile[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	if (xIndex < width && yIndex < height)
	{
		odata[index_out] = tile[threadIdx.y][threadIdx.x];
	}
}

// computes caloric cost given a vector representing the direction traveled (including elevation change)
// this should be identical to the caloric cost function used by the fast marching algorithm on the CPU
__device__ float caloricCostFunc(float3 travelVec, float terrainFactor, char tSex, float tSpeed, float tAge, float tWeight, float tHeight, float lWeight)
{
	float euclidDistance;
	float rise;
	float run;
	float percentGrade;
	float LOverWSquared;
	float VSquared;
	float GPlus6Squared;
	float WPlusL;
	float wattsPE;
	float wattsCF;
	float watts;
	float minutesRequired;
	float caloriesPerMinute;
	float kgcalories;
	float basalKcalPerMinute;
	float basalKcal;

	euclidDistance = sqrtf((travelVec.x * travelVec.x) + (travelVec.y * travelVec.y) + (travelVec.z * travelVec.z));
	rise = travelVec.y;
	travelVec.y = 0.0;
	run = sqrtf((travelVec.x * travelVec.x) + (travelVec.z * travelVec.z));
	percentGrade = (rise / run) * 100.0;

	LOverWSquared = powf((lWeight / tWeight), 2.0);
	VSquared = tSpeed * tSpeed;
	GPlus6Squared = powf((percentGrade + 6.0), 2.0);
	WPlusL = tWeight + lWeight;
	wattsPE = 1.5 * tWeight + 2.0 * WPlusL * LOverWSquared
		+ terrainFactor * WPlusL * (1.5 * VSquared + 0.35 * tSpeed * percentGrade);
	wattsCF = terrainFactor * ((percentGrade * WPlusL * tSpeed) / 3.5
		- (WPlusL * GPlus6Squared / tWeight) + (25.0 - VSquared));

	if (percentGrade >= 0.0)
	{
		watts = wattsPE;
	}
	else
	{
		watts = wattsPE - wattsCF;
	}

	minutesRequired = euclidDistance / (tSpeed * 60.0);
	caloriesPerMinute = watts * 0.01433;
	kgcalories = caloriesPerMinute * minutesRequired;

	/* Harris-Benedict equation for Basal Metabolism (Kcals per day)
	* Males: 66 + (13.7 * WeightKg) + (5 * HeightCM) - (6.8* Age)
	* Females: 655 + (9.6 * WeightKg) + (1.7 * HeightCM) - (4.7 * Age) */
	if (tSex)
	{
		basalKcalPerMinute = (66.0 + (13.7 * tWeight) + (5.0 * tHeight) - (6.8 * tAge)) / 1440.0;
	}
	else
	{
		basalKcalPerMinute = (655.0 + (9.6 * tWeight) + (1.7 * tHeight) - (4.7 * tAge)) / 1440.0;
	}
	basalKcal = basalKcalPerMinute * minutesRequired;

	if (kgcalories < basalKcal)
	{
		kgcalories = basalKcal;
	}

	return kgcalories;
}

// calculates an approximation of the caloric "wavefront" propigating forward into a given data point
// given the caloric value and travel cost of its neighbors
__device__ float propFunc(float xWork, float xCost, float zWork, float zCost)
{
	float solution = -1.0;

	if (xWork >= 0.0 && zWork >= 0.0)
	{
		float a = (1.0 / (xCost * xCost)) + (1.0 / (zCost * zCost));
		float b = ((-2.0 * xWork) / (xCost * xCost)) + ((-2.0 * zWork) / (zCost * zCost));
		float c = ((xWork * xWork) / (xCost * xCost)) + ((zWork * zWork) / (zCost * zCost)) - 1.0;

		float rad = (b * b) + (-4.0 * a * c);

		if (rad < 0.0)
		{
			solution = (-1.0 * b) / (2.0 * a);
		}
		else
		{
			float solution1 = ((-1.0 * b) + sqrtf(rad)) / (2.0 * a);
			float solution2 = ((-1.0 * b) - sqrtf(rad)) / (2.0 * a);

			if (solution1 > solution2)
			{
				solution = solution1;
			}
			else
			{
				solution = solution2;
			}
		}
	}
	else if (xWork >= 0.0)
	{
		float a = (1.0 / (xCost * xCost));
		float b = ((-2.0 * xWork) / (xCost * xCost));
		float c = ((xWork * xWork) / (xCost * xCost)) - 1.0;

		float rad = (b * b) + (-4.0 * a * c);

		if (rad < 0.0)
		{
			solution = (-1.0 * b) / (2.0 * a);
		}
		else
		{
			float solution1 = ((-1.0 * b) + sqrtf(rad)) / (2.0 * a);
			float solution2 = ((-1.0 * b) - sqrtf(rad)) / (2.0 * a);

			if (solution1 > solution2)
			{
				solution = solution1;
			}
			else
			{
				solution = solution2;
			}
		}
	}
	else if (zWork >= 0.0)
	{
		float a = (1.0 / (zCost * zCost));
		float b = ((-2.0 * zWork) / (zCost * zCost));
		float c = ((zWork * zWork) / (zCost * zCost)) - 1.0;

		float rad = (b * b) + (-4.0 * a * c);

		if (rad < 0.0)
		{
			solution = (-1.0 * b) / (2.0 * a);
		}
		else
		{
			float solution1 = ((-1.0 * b) + sqrtf(rad)) / (2.0 * a);
			float solution2 = ((-1.0 * b) - sqrtf(rad)) / (2.0 * a);

			if (solution1 > solution2)
			{
				solution = solution1;
			}
			else
			{
				solution = solution2;
			}
		}
	}

	return solution;
}

#define ELEVATION_AT(idxX, idxZ)  (rotated ? tex2D(demTex, (idxZ), (idxX)) : tex2D(demTex, (idxX), (idxZ)))
#define CALORIES_AT(idxX, idxZ)  (((idxX) >= 0 && (idxX) < demWidth && (idxZ) >= 0 && (idxZ) < demDepth) ? workCals[((idxZ) * demWidth) + (idxX)] : -1)
#define CALORIES_AT_LAST(idxX)  (((idxX)+1 >= 0 && (idxX)+1 < THREAD_COUNT+2) ? lastRow[(idxX)+1] : -1)
#define CALORIES_AT_CUR(idxX)  (((idxX)+1 >= 0 && (idxX)+1 < THREAD_COUNT+2) ? curRow[(idxX)+1] : -1)

// a kernel for running a parallel raster scan technique to calculate caloric cost over a elevation data set.
// Based off of "Parallel algorithms for approximation of distance maps on parametric surfaces"
// http://visl.technion.ac.il/bron/publications/WebDevBroBroKimTOG08.pdf
__global__ void rsKernel(float* workCals, int* updated, char rotated, int row, int demWidth, int demDepth, float cellSize)
{
	__shared__ float lastRow[THREAD_COUNT+2];
	__shared__ float curRow[THREAD_COUNT+2];

	// take into account the overlap
	int blockStartX = (blockIdx.x * THREAD_COUNT) - (blockIdx.x * STRIP_HEIGHT * 2);
	int blockSafeX = blockStartX + THREAD_COUNT - (2 * STRIP_HEIGHT);

	int kernelX = blockStartX + threadIdx.x;
	float3 travelVec;

	lastRow[threadIdx.x+1] = CALORIES_AT(kernelX, row - 1);
	curRow[threadIdx.x+1] = CALORIES_AT(kernelX, row);
	if (threadIdx.x == 0)
	{
		lastRow[0] = CALORIES_AT(kernelX-1, row - 1);
		curRow[0] = CALORIES_AT(kernelX-1, row);
	}
	else if (threadIdx.x == THREAD_COUNT - 1)
	{
		lastRow[THREAD_COUNT+1] = CALORIES_AT(kernelX+1, row - 1);
		curRow[THREAD_COUNT+1] = CALORIES_AT(kernelX+1, row);
	}

	__syncthreads();

	// work on each row in the strip for the top down direction
	for (int i = 0; i < STRIP_HEIGHT; i++)
	{
		float solution = -1.0;
		int topDownZ = row + i;

		if (topDownZ < demDepth && kernelX < demWidth)
		{
			float leftWork = CALORIES_AT_CUR(((int)threadIdx.x) - 1);
			float upWork = CALORIES_AT_LAST(((int)threadIdx.x));
			float rightWork = CALORIES_AT_CUR(((int)threadIdx.x) + 1);
			float leftCost = -1.0;
			float upCost = -1.0;
			float rightCost = -1.0;

			if (upWork >= 0.0)
			{
				travelVec.x = 0.0;
				travelVec.y = ELEVATION_AT(kernelX, topDownZ) - ELEVATION_AT(kernelX, topDownZ - 1);
				travelVec.z = cellSize;
				upCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}
			if (leftWork >= 0.0)
			{
				travelVec.x = cellSize;
				travelVec.y = ELEVATION_AT(kernelX, topDownZ) - ELEVATION_AT(kernelX - 1, topDownZ);
				travelVec.z = 0.0;
				leftCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}
			if (rightWork >= 0.0)
			{
				travelVec.x = cellSize;
				travelVec.y = ELEVATION_AT(kernelX, topDownZ) - ELEVATION_AT(kernelX + 1, topDownZ);
				travelVec.z = 0.0;
				rightCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}

			float zWork = upWork;
			float zCost = upCost;
			float xWork = leftWork;
			float xCost = leftCost;
			if (leftWork < 0.0 || (rightWork >= 0.0 && rightWork < leftWork))
			{
				xWork = rightWork;
				xCost = rightCost;
			}
			solution = propFunc(xWork, xCost, zWork, zCost);
		}

		__syncthreads();

		if (solution >= 0.0 && kernelX < blockSafeX && (curRow[threadIdx.x] < 0.0 || solution < curRow[threadIdx.x]))
		{
			workCals[(topDownZ * demWidth) + kernelX] = solution;
			curRow[threadIdx.x + 1] = solution;
			*updated = 1;
		}

		lastRow[threadIdx.x] = curRow[threadIdx.x];
		curRow[threadIdx.x] = CALORIES_AT(kernelX, topDownZ + 1);
		if (threadIdx.x == 0)
		{
			curRow[0] = CALORIES_AT(kernelX - 1, topDownZ + 1);
		}
		else if (threadIdx.x == THREAD_COUNT - 1)
		{
			curRow[THREAD_COUNT + 1] = CALORIES_AT(kernelX + 1, topDownZ + 1);
		}

		__syncthreads();
	}

	curRow[threadIdx.x] = CALORIES_AT(kernelX, (demDepth - row - 1));
	lastRow[threadIdx.x] = CALORIES_AT(kernelX, (demDepth - row - 1) + 1);
	if (threadIdx.x == 0)
	{
		curRow[0] = CALORIES_AT(kernelX - 1, (demDepth - row - 1));
		lastRow[0] = CALORIES_AT(kernelX - 1, (demDepth - row - 1) + 1);
	}
	else if (threadIdx.x == THREAD_COUNT - 1)
	{
		curRow[THREAD_COUNT + 1] = CALORIES_AT(kernelX + 1, (demDepth - row - 1));
		lastRow[THREAD_COUNT + 1] = CALORIES_AT(kernelX + 1, (demDepth - row - 1) + 1);
	}

	__syncthreads();

	// work on each row in the strip for the bottom up direction
	for (int i = 0; i < STRIP_HEIGHT; i++)
	{
		float solution = -1.0;
		int bottomUpZ = (demDepth - row - 1) - i;

		if (bottomUpZ >= 0 && kernelX < demWidth)
		{
			float leftWork = CALORIES_AT_CUR(((int)threadIdx.x) - 1);
			float downWork = CALORIES_AT_LAST(((int)threadIdx.x));
			float rightWork = CALORIES_AT_CUR(((int)threadIdx.x) + 1);
			float leftCost = -1.0;
			float downCost = -1.0;
			float rightCost = -1.0;

			if (downWork >= 0.0)
			{
				travelVec.x = 0.0;
				travelVec.y = ELEVATION_AT(kernelX, bottomUpZ) - ELEVATION_AT(kernelX, bottomUpZ + 1);
				travelVec.z = cellSize;
				downCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}
			if (leftWork >= 0.0)
			{
				travelVec.x = cellSize;
				travelVec.y = ELEVATION_AT(kernelX, bottomUpZ) - ELEVATION_AT(kernelX - 1, bottomUpZ);
				travelVec.z = 0.0;
				leftCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}
			if (rightWork >= 0.0)
			{
				travelVec.x = cellSize;
				travelVec.y = ELEVATION_AT(kernelX, bottomUpZ) - ELEVATION_AT(kernelX + 1, bottomUpZ);
				travelVec.z = 0.0;
				rightCost = caloricCostFunc(travelVec, 1.0, 1, 1.25, 30.0, 80.0, 183.0, 0.0); // change this to not be constants
			}

			float zWork = downWork;
			float zCost = downCost;
			float xWork = leftWork;
			float xCost = leftCost;
			if (leftWork < 0.0 || (rightWork >= 0.0 && rightWork < leftWork))
			{
				xWork = rightWork;
				xCost = rightCost;
			}
			solution = propFunc(xWork, xCost, zWork, zCost);
		}

		__syncthreads();

		if (solution >= 0.0 && kernelX < blockSafeX && (curRow[threadIdx.x] < 0.0 || solution < curRow[threadIdx.x]))
		{
			workCals[(bottomUpZ * demWidth) + kernelX] = solution;
			curRow[threadIdx.x + 1] = solution;
			*updated = 1;
		}

		lastRow[threadIdx.x] = curRow[threadIdx.x];
		curRow[threadIdx.x] = CALORIES_AT(kernelX, bottomUpZ - 1);
		if (threadIdx.x == 0)
		{
			curRow[0] = CALORIES_AT(kernelX - 1, bottomUpZ - 1);
		}
		else if (threadIdx.x == THREAD_COUNT - 1)
		{
			curRow[THREAD_COUNT + 1] = CALORIES_AT(kernelX + 1, bottomUpZ - 1);
		}

		__syncthreads();
	}
}

int h_updated;
cudaArray* d_demElevs;
float* d_workCals;
float* d_workCalsT;
int* d_updated;
int blockCountUpDown;
int blockCountLeftRight;
dim3 tgrid, tgridt;
dim3 tthreads(TILE_DIM, TILE_DIM);

// called exernally to initialize the data used by this algorithm
extern "C"
void rasterScanCudaInit()
{
	h_updated = 1;

	demTex.addressMode[0] = cudaAddressModeBorder;
	demTex.addressMode[1] = cudaAddressModeBorder;
	demTex.filterMode = cudaFilterModePoint;
	demTex.normalized = false;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// expecting things to have been started already by fast marching on the CPU
	//workCals[(startz * demHeader.width) + startx] = 0.0;

	checkCudaErrors(cudaMalloc((void **)&d_workCals, demHeader.width * demHeader.depth * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_workCalsT, demHeader.width * demHeader.depth * sizeof(float)));
	checkCudaErrors(cudaMallocArray(&d_demElevs, &channelDesc, demHeader.width, demHeader.depth));
	checkCudaErrors(cudaMalloc((void **)&d_updated, sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_workCals, workCals, demHeader.width * demHeader.depth * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(d_demElevs, 0, 0, demElevs, demHeader.width * demHeader.depth * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTextureToArray(demTex, d_demElevs, channelDesc));

	int blockSafeZone = (THREAD_COUNT - (STRIP_HEIGHT * 2));
	blockCountUpDown = demHeader.width / blockSafeZone;
	blockCountLeftRight = demHeader.depth / blockSafeZone;
	if (demHeader.width % blockSafeZone != 0)
		blockCountUpDown++;
	if (demHeader.depth % blockSafeZone != 0)
		blockCountLeftRight++;

	tgrid.x = demHeader.width / TILE_DIM;
	tgrid.y = demHeader.depth / TILE_DIM;
	tgridt.x = demHeader.depth / TILE_DIM;
	tgridt.y = demHeader.width / TILE_DIM;
	if (demHeader.width % TILE_DIM != 0)
	{
		tgrid.x += 1;
		tgridt.y += 1;
	}
	if (demHeader.depth % TILE_DIM != 0)
	{
		tgrid.y += 1;
		tgridt.x += 1;
	}

	printf("Running raster scan algorithm on GPU to compute caloric costs.\n");
	printf("For Up/Down using %d blocks of %d threads and a strip height of %d\n", blockCountUpDown, THREAD_COUNT, STRIP_HEIGHT);
	printf("For Left/Right using %d blocks of %d threads and a strip height of %d\n", blockCountLeftRight, THREAD_COUNT, STRIP_HEIGHT);
	printf("For transpose to using %u X %u blocks of %u X %u threads\n", tgrid.x, tgrid.y, tthreads.x, tthreads.y);
	printf("For transpose from using %u X %u blocks of %u X %u threads\n", tgridt.x, tgridt.y, tthreads.x, tthreads.y);
}

// called externally to cleanup after the algorithm
extern "C"
void rasterScanCudaCleanup()
{
	checkCudaErrors(cudaFreeArray(d_demElevs));
	checkCudaErrors(cudaFree(d_workCals));

	checkCudaErrors(cudaDeviceReset());
}

// "step" the algorithm by running raster scan once in each of the four directions.
extern "C"
char rasterScanCudaStep(float terrainFactor, char tSex, float tSpeed, float tAge, float tWeight, float tHeight, float lWeight)
{
	if (h_updated)
	{
		h_updated = 0;
		checkCudaErrors(cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice));

		// do up and down
		for (int i = 0; i < demHeader.depth; i += STRIP_HEIGHT)
		{
			rsKernel<<<blockCountUpDown, THREAD_COUNT>>>(d_workCals, d_updated, 0, i, demHeader.width, demHeader.depth, demHeader.cellSize);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}
		
		// transpose to do left/right as up/down
		transpose<<<tgrid, tthreads>>>(d_workCalsT, d_workCals, demHeader.width, demHeader.depth);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// do left and right
		for (int i = 0; i < demHeader.width; i += STRIP_HEIGHT)
		{
			rsKernel<<<blockCountLeftRight, THREAD_COUNT>>>(d_workCalsT, d_updated, 1, i, demHeader.depth, demHeader.width, demHeader.cellSize);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		// transpose back
		transpose<<<tgridt, tthreads>>>(d_workCals, d_workCalsT, demHeader.depth, demHeader.width);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(workCals, d_workCals, demHeader.width * demHeader.depth * sizeof(float), cudaMemcpyDeviceToHost));
		
		return 0;
	}
	else
	{
		printf("\nDone with raster scan on the GPU\n");

		rasterScanCudaCleanup();

		return 1;
	}
}

// run raster scan until it converges on a solution
extern "C" void rasterScanCuda(float terrainFactor, char tSex, float tSpeed, float tAge, float tWeight, float tHeight, float lWeight)
{
	int loopCount = 0;

	while (h_updated && loopCount < 10000)
	{
		h_updated = 0;
		checkCudaErrors(cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice));
		
		// do up and down
		for (int i = 0; i < demHeader.depth; i += STRIP_HEIGHT)
		{
			rsKernel<<<blockCountUpDown, THREAD_COUNT>>>(d_workCals, d_updated, 0, i, demHeader.width, demHeader.depth, demHeader.cellSize);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		// transpose to do left/right as up/down
		transpose<<<tgrid, tthreads>>>(d_workCalsT, d_workCals, demHeader.width, demHeader.depth);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// do left and right
		for (int i = 0; i < demHeader.width; i += STRIP_HEIGHT)
		{
			rsKernel<<<blockCountLeftRight, THREAD_COUNT>>>(d_workCalsT, d_updated, 1, i, demHeader.depth, demHeader.width, demHeader.cellSize);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		// transpose back
		transpose<<<tgridt, tthreads>>>(d_workCals, d_workCalsT, demHeader.depth, demHeader.width);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


		checkCudaErrors(cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost));

		printf(".");
		fflush(stdout);
		loopCount++;
	}

	printf("\nDone with raster scan on the GPU after %d iterations\n", loopCount);

	checkCudaErrors(cudaMemcpy(workCals, d_workCals, demHeader.width * demHeader.depth * sizeof(float), cudaMemcpyDeviceToHost));

	rasterScanCudaCleanup();
}

#endif
