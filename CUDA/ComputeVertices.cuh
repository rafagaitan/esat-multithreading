#pragma once

#include "cuda_runtime.h"

const unsigned int meshWidth = 512;
const unsigned int meshHeight = 512;
const unsigned int NUM_VERTICES = meshWidth*meshHeight;


void computeVertices(float4* pos, unsigned int width, unsigned int height, float time);

