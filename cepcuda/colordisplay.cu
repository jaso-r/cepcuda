#ifndef __ceptools_colordisplay_cu
#define __ceptools_colordisplay_cu

/*
 *  colordisplay.cu
 *
 *  CUDA code used to modify the texture image displaying algorithm results.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include "cepdb.h"
#include "globals.h"

#define THREAD_CNT 512

texture<float, 2> elevTex;
cudaArray *d_elev_array;

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

// kernel sets the "color" value based off of caloric cost if it has been calculated or elevation otherwise.
__global__ void color_kernel(float *od, float *d_cals, int len, int width, float minElevation, float maxElevation, float bandSize)
{
	int arrayIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	int xIndex = arrayIndex % width;
	int yIndex = arrayIndex / width;

	if (arrayIndex < len)
	{
		float cal = d_cals[arrayIndex];
		if (cal >= 0)
		{
			od[arrayIndex] = ((float)(((int)cal) % ((int)bandSize))) / bandSize;
		}
		else
		{
			od[arrayIndex] = (tex2D(elevTex, xIndex, yIndex) - minElevation) / (maxElevation - minElevation);
		}
	}
}

// called externally to initialize the textures used by the kernel
extern "C"
void initTexture(int width, int height, void *elevVals)
{
	int size = width * height * sizeof(float);

	// copy image data to array
	cudaChannelFormatDesc channelDesc;
	channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	checkCudaErrors(cudaMallocArray(&d_elev_array, &channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(d_elev_array, 0, 0, elevVals, size, cudaMemcpyHostToDevice));

	// set texture parameters
	elevTex.addressMode[0] = cudaAddressModeClamp;
	elevTex.addressMode[1] = cudaAddressModeClamp;
	elevTex.filterMode = cudaFilterModePoint;
	elevTex.normalized = false;

	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(elevTex, d_elev_array, channelDesc));
}

// called externally to cleanup the textures used by the kernel
extern "C"
void freeTextures()
{
	checkCudaErrors(cudaFreeArray(d_elev_array));
}

// called externally to run the kernel and update the texture image based off of the currently calculated calorie data
// and the elevation data.
extern "C"
void convertToColor(float *d_dest, float *h_cals, int width, int height, float minElevation, float maxElevation, float bandSize)
{
	// sync host and start computation timer_kernel
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaBindTextureToArray(elevTex, d_elev_array));

	float *d_cals = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_cals, width * height * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_cals, h_cals, width * height * sizeof(float), cudaMemcpyHostToDevice));

	int blockCount = (width * height) / THREAD_CNT;
	if ((width * height) % THREAD_CNT != 0) blockCount++;

	color_kernel<<<blockCount, THREAD_CNT>>>(d_dest, d_cals, (width * height), width, minElevation, maxElevation, bandSize);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_cals));
}


#endif