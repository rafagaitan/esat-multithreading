/*******************************************************************************
   The MIT License (MIT)

   Copyright (c) 2014 Rafael Gaitan <rafa.gaitan@mirage-tech.com>
                                    http://www.mirage-tech.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.

   -----------------------------------------------------------------------------
   Additional Notes:

   Code for the Multithreading and Parallel Computing Course at ESAT
               -------------------------------
               |     http://www.esat.es      |
               -------------------------------

   more information of the course at:
       -----------------------------------------------------------------
       |  http://www.esat.es/estudios/programacion-multihilo/?pnt=621  |
       -----------------------------------------------------------------
**********************************************************************************/
// session 8
__kernel void multKernel(__global int *c, __global const int *a, __global const int *b)
{
    int i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void square(__global float* input, __global float* output)
{
    int i = get_global_id(0);
    output[i] = input[i]*input[i];
}

__kernel void dp_mul(__global const float *a,
                     __global const float *b,
                     __global float *c,
                     int N)
{
    int id = get_global_id(0);
    if (id < N)
        c[id] = a[id] * b[id];
}

__kernel void addKernel(__global float *c, __global const float *a, __global const float *b)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}

__kernel void fillKernel(__global float *a, __global float* b)
{
    int i = get_global_id(0);
    a[i] = sin((float)i)*sin((float)i);
    b[i] = cos((float)i)*cos((float)i);
}

__kernel void fillAndAddKernel(__global float* c, __global float *a, __global float* b)
{
    int i = get_global_id(0);
    a[i] = sin((float)i)*sin((float)i);
    b[i] = cos((float)i)*cos((float)i);
    c[i] = a[i] + b[i];
}


// session 10

__kernel void computeVertices(__global float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0 - 1.0f;
    v = v*2.0 - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = (float4)(u, w, v, 1.0f);
}
