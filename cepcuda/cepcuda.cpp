/**
 * cepcuda
 * Implements both a CPU and CUDA based algorithm for computing least-Caloric-cost geodesics over digital
 * elevation data sets and displays the results of those algorithms.  The CPU algorithm is a fast marching
 * approach and the parallel CUDA algorithm is more akin to a raster-scan approach.
 */

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// standard includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "makedb.h"
#include "fastmarchingcpu.h"

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10
#define REFRESH_DELAY     100 //ms

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

// program control inputs
char doCpu = 0;
int startX = -1;
int startZ = -1;
char novisualize = 0;
char algorithmDone = 0;

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint texid;   // Texture
GLuint shader;
float *h_img = NULL;


// externs to the functions defined in the *.cu files
extern "C" void rasterScanCuda(float terrainFactor, char tSex, float tSpeed, float tAge, float tWeight, float tHeight, float lWeight);
extern "C" char rasterScanCudaStep(float terrainFactor, char tSex, float tSpeed, float tAge, float tWeight, float tHeight, float lWeight);
extern "C" void rasterScanCudaInit();
extern "C" void rasterScanCudaCleanup();
extern "C" void initTexture(int width, int height, void *elevVals);
extern "C" void freeTextures();
extern "C" void convertToColor(float *d_dest, float *h_cals, int width, int height, float minElevation, float maxElevation, float bandSize);


void printUsageAndExit()
{
	printf("cepcuda <demfile> [-cpu] [-novisualize] [-start x z] [-s m|f] [-v mps] [-a age] [-w kg] [-h cm] [-l kg]\n");
	printf("\tPerforms fast marching on the DEM data.\n");
	printf("\t<demfile> the name of the dem file to perform fast marching on. This must be the first parameter.\n");
	printf("\t[-cpu] optional flag to indicate that fast marching should be performed on the CPU instead of the GPU\n");
	printf("\t[-novisualize] optional flag to indicate that instead of visualizing progress of the algorithms, just calculate offline and show the result.\n");
	printf("\t[-start x z] optional parameter to set the start point for fast marching. Defaults to the middle of the DEM data. Takes integer coords.\n");
	printf("\t[-s m|f] optional parameter to set the sex of the traveler. Must be 'm' for male or 'f' for female. Defaults to male.\n");
	printf("\t[-v mps] optional parameter to set the traveler's velocity in meters per second. Defaults to 1.25 (about 4.5 km/hr).\n");
	printf("\t[-a age] optional parameter to set the age of the traveler. Defaults to 30.\n");
	printf("\t[-w kg] optional parameter to set the weight of the traveler in kilograms. Defaults to 80 (about 175 lbs).\n");
	printf("\t[-h cm] optional parameter to set the height of the traveler in centimeters. Defaults to 183 (about 6 ft).\n");
	printf("\t[-l kg] optional parameter to set the weight of the traveler's load in kilograms. Defaults to 0.\n");
	exit(0);
}

void parseCommandLineArguments(int argc, char * argv[])
{
	int idx = 2;
	while (idx < argc)
	{
		if (strcmp("-s", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			if (strcmp("m", argv[idx]) == 0)
			{
				travelerSex = 1;
			}
			else if (strcmp("f", argv[idx]) == 0)
			{
				travelerSex = 0;
			}
			else
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-start", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			startX = atoi(argv[idx]);
			if (startX < 0)
			{
				printUsageAndExit();
			}
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			startZ = atoi(argv[idx]);
			if (startZ < 0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-v", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			travelerSpeed = atof(argv[idx]);
			if (travelerSpeed == 0.0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-a", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			travelerAge = atof(argv[idx]);
			if (travelerAge == 0.0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-w", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			travelerWeight = atof(argv[idx]);
			if (travelerWeight == 0.0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-h", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			travelerHeight = atof(argv[idx]);
			if (travelerHeight == 0.0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-l", argv[idx]) == 0)
		{
			idx++;
			if (idx == argc)
			{
				printUsageAndExit();
			}
			loadWeight = atof(argv[idx]);
			if (loadWeight == 0.0)
			{
				printUsageAndExit();
			}
		}
		else if (strcmp("-cpu", argv[idx]) == 0)
		{
			doCpu = 1;
		}
		else if (strcmp("-novisualize", argv[idx]) == 0)
		{
			novisualize = 1;
		}
		else
		{
			printUsageAndExit();
		}
		idx++;
	}
}

void printVariables()
{
	printf("Caloric cost variables:\n");
	printf("\tTraveler sex: ");
	if (travelerSex)
	{
		printf("male\n");
	}
	else
	{
		printf("female\n");
	}
	printf("\tTraveler speed: %f meters per second\n", (float)travelerSpeed);
	printf("\tTraveler age: %f years old\n", (float)travelerAge);
	printf("\tTraveler height: %f centimeters\n", (float)travelerHeight);
	printf("\tTraveler weight: %f kilograms\n", (float)travelerWeight);
	printf("\tLoad weight: %f kilograms\n", (float)loadWeight);
}

// perform the setup necessary for the selected algorithm
void algorithmSetup()
{
	workCals = (float*)malloc(sizeof(float) * demHeader.width * demHeader.depth);

	if (doCpu)
	{
		workDatas = (DEMWorkData*)malloc(sizeof(DEMWorkData) * demHeader.width * demHeader.depth);
		for (int i = 0; i < (demHeader.width * demHeader.depth); i++)
		{
			workDatas[i].calories = -1.0;
			workDatas[i].visited = 0;
			workDatas[i].heapIndex = 0;
			workDatas[i].propagation = 0;
		}

		fastMarchingInit(startX, startZ);
	}
	else
	{
		workDatas = (DEMWorkData*)malloc(sizeof(DEMWorkData) * demHeader.width * demHeader.depth);
		for (int i = 0; i < (demHeader.width * demHeader.depth); i++)
		{
			workDatas[i].calories = -1.0;
			workDatas[i].visited = 0;
			workDatas[i].heapIndex = 0;
			workDatas[i].propagation = 0;
		}

		fastMarchingInit(startX, startZ);
		fastMarching(1024);

		demElevs = (float*)malloc(sizeof(float) * demHeader.width * demHeader.depth);
		for (int i = 0; i < (demHeader.width * demHeader.depth); i++)
		{
			workCals[i] = workDatas[i].calories;
			demElevs[i] = demData.elevation[i];
		}
		free(workDatas);

		rasterScanCudaInit();
	}
}

// step the selected algorithm by some amount
char stepAlgorithm()
{
	printf(".");
	fflush(stdout);

	if (doCpu)
	{
		char ret = fastMarchingStep();
		for (int i = 0; i < (demHeader.width + demHeader.depth) / 2 && ret == 0; i++)
		{
			ret = fastMarchingStep();
		}
		return ret;
	}
	else
	{
		return rasterScanCudaStep(1.0, travelerSex, travelerSpeed, travelerAge, travelerWeight, travelerHeight, loadWeight);
	}
}

// runs the selected algorithm from start to finish
void runAlgorithm()
{
	time_t startTime;
	time_t endTime;

	if (doCpu)
	{
		startTime = time(NULL);
		fastMarching(-1);
		endTime = time(NULL);

		printf("Corner cost: %f\n", workDatas[0].calories);
		printf("Greatest calories: %f\n", greatestCalories);
	}
	else
	{
		startTime = time(NULL);
		rasterScanCuda(1.0, travelerSex, travelerSpeed, travelerAge, travelerWeight, travelerHeight, loadWeight);
		endTime = time(NULL);

		printf("Corner cost: %f\n", workCals[0]);
	}

	printf("Run time: %d\n", (int)(endTime - startTime));
}

// display results using OpenGL
void display()
{
	// execute filter, writing results to pbo
	float *d_result;
	
	if (!novisualize && !algorithmDone)
	{
		algorithmDone = stepAlgorithm();
	}

	if (doCpu)
	{
		for (int i = 0; i < demHeader.depth; i++)
		{
			for (int j = 0; j < demHeader.width; j++)
			{
				workCals[(i * demHeader.width) + j] = workDatas[(i * demHeader.width) + j].calories;
			}
		}
	}

	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes, cuda_pbo_resource));
	
	convertToColor(d_result, workCals, demHeader.width, demHeader.depth, demHeader.minElevation, demHeader.maxElevation, 750.0);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	// OpenGL display code path
	{
		glClear(GL_COLOR_BUFFER_BIT);

		// load texture from pbo
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, demHeader.width, demHeader.depth, GL_RED, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// fragment program is required to display floating point texture
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
		glEnable(GL_FRAGMENT_PROGRAM_ARB);
		glDisable(GL_DEPTH_TEST);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f);
			glVertex2f(0.0f, 0.0f);
			glTexCoord2f(1.0f, 0.0f);
			glVertex2f(1.0f, 0.0f);
			glTexCoord2f(1.0f, 1.0f);
			glVertex2f(1.0f, 1.0f);
			glTexCoord2f(0.0f, 1.0f);
			glVertex2f(0.0f, 1.0f);
		}
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_FRAGMENT_PROGRAM_ARB);
	}

	glutSwapBuffers();
	glutReportErrors();
}

// Timer Event so we can refresh the display
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

// Resizing the window
void reshape(int x, int y)
{
	glViewport(0, 0, x, y);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

// setup the visualization texture
void initCuda()
{
	initTexture(demHeader.width, demHeader.depth, demData.elevation);
}

// cleanup everything used in the display as well as cleanup
// for the selected algorithm
void cleanup()
{
	freeTextures();

	cudaGraphicsUnregisterResource(cuda_pbo_resource);

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &texid);
	glDeleteProgramsARB(1, &shader);

	if (doCpu)
	{
		fastMarchingCleanup();
	}
	else
	{
		rasterScanCudaCleanup();
	}

	if (h_img)
	{
		free(h_img);
		h_img = NULL;
	}
}

// shader for displaying floating-point texture
static const char *shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		printf("Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

// This is where we create the OpenGL PBOs, FBOs, and texture resources
void initGLResources()
{
	// create pixel buffer object
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, demHeader.width * demHeader.depth * sizeof(float), h_img, GL_STREAM_DRAW_ARB);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, demHeader.width, demHeader.depth, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	// load shader program
	shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void initGL(int *argc, char **argv)
{
	// initialize GLUT
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(768, 768);
	glutCreateWindow("Caloric Cost Geodesics");
	glutDisplayFunc(display);

	//glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	glewInit();

	if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
	{
		printf("Error: failed to get minimal extensions for demo\n");
		printf("This sample requires:\n");
		printf("  OpenGL version 1.5\n");
		printf("  GL_ARB_vertex_buffer_object\n");
		printf("  GL_ARB_pixel_buffer_object\n");
		exit(EXIT_FAILURE);
	}
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
	int runtimeVersion = 0;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	cudaRuntimeGetVersion(&runtimeVersion);
	fprintf(stderr, "  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	fprintf(stderr, "  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

	if (runtimeVersion >= min_runtime && ((deviceProp.major << 4) + deviceProp.minor) >= min_compute)
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findCapableDevice(int argc, char **argv)
{
	int dev;
	int bestDev = -1;

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0)
	{
		fprintf(stderr, "There are no CUDA capabile devices.\n");
		exit(EXIT_SUCCESS);
	}
	else
	{
		fprintf(stderr, "Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);
	}

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		if (checkCUDAProfile(dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION))
		{
			fprintf(stderr, "\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name);

			if (bestDev == -1)
			{
				bestDev = dev;
				fprintf(stderr, "Setting active device to %d\n", bestDev);
			}
		}
	}

	if (bestDev == -1)
	{
		fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
		fprintf(stderr, "The CUDA Sample minimum requirements:\n");
		fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION / 16, MIN_COMPUTE_VERSION % 16);
		fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION / 1000, (MIN_RUNTIME_VERSION % 100) / 10);
		exit(EXIT_SUCCESS);
	}

	return bestDev;
}

int main(int argc, char * argv[])
{
	FILE* demFile;

	if (argc < 2)
	{
		printUsageAndExit();
	}
	else
	{
		demFile = fopen(argv[1], "r");
	}

	if (demFile == NULL)
	{
		printf("Error opening dem file\n");
		return 1;
	}

	if (argc > 2)
	{
		parseCommandLineArguments(argc, argv);
	}

	printVariables();

	// load the digital elevation data
	loadDEMData(&demHeader, &demData, demFile);
	fclose(demFile);

	if (startX < 0.0)
	{
		startX = demHeader.width / 2;
		startZ = demHeader.depth / 2;
	}

	// create an initial display image based off of the elevation data
	h_img = (float*) malloc(demHeader.width * demHeader.depth * sizeof(float));
	for (int i = 0; i < demHeader.depth; i++)
	{
		for (int j = 0; j < demHeader.width; j++)
		{
			h_img[(i * demHeader.width) + j] = (demData.elevation[(i * demHeader.width) + j] - demHeader.minElevation) / (demHeader.maxElevation - demHeader.minElevation);
		}
	}

	// initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);
	int dev = findCapableDevice(argc, argv);

	if (dev != -1)
	{
		cudaGLSetGLDevice(dev);
	}
	else
	{
		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	// Now we can create a CUDA context and bind it to the OpenGL context
	initCuda();
	initGLResources();

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	algorithmSetup();
	
	// if we aren't visualizing the algorithm steps, then just run the whole thing now
	if (novisualize)
	{
		runAlgorithm();
	}

	// start the loop
	glutMainLoop();

	return 0;
}
