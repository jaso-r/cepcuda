#ifndef __ceptools_globals_h
#define __ceptools_globals_h

#include "cepdb.h"

// these are global to the program so that I could just rip off a bunch of
// my old code that had these global
extern char travelerSex;
extern double travelerSpeed;
extern double travelerAge;
extern double travelerWeight;
extern double travelerHeight;
extern double loadWeight;
extern double greatestCalories;

extern DEMHeader demHeader;
extern DEMElevationData demData;
extern DEMWorkData* workDatas;
extern float* workCals;
extern float* demElevs;

#endif
