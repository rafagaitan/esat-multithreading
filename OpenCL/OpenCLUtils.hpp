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
#ifndef _OPENCL_UTILS_HPP_
#define _OPENCL_UTILS_HPP_

#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>

#include "Config.hpp"

//#include <CL/cl.h>
#include "cl.hpp"

namespace opencl
{
std::string loadKernel (const char* name)
{
    std::ifstream in (name);
    if (!in.good())
    {
        in.open((std::string(RESOURCES_PATH) + std::string("/") + std::string(name)).c_str());
    }
    if (!in.good())
    {
        std::cerr << "Error loading kernel:" << name << std::endl;
        return std::string();
    }
    std::string result (
        (std::istreambuf_iterator<char> (in)),
        std::istreambuf_iterator<char> ());
    return result;
}

cl_context_properties getPlaformIdForType(cl_device_type type)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl_int error = CL_SUCCESS;
    // Check the platforms we found for a device of our specified type
    cl_context_properties platform_id = 0;
    for (unsigned int i = 0; i < platforms.size(); i++) {

        std::vector<cl::Device> devices;

#if defined(__CL_ENABLE_EXCEPTIONS)
        try {
#endif

            error = platforms[i].getDevices(type, &devices);

#if defined(__CL_ENABLE_EXCEPTIONS)
        }
        catch (cl::Error) {}
        // Catch if exceptions are enabled as we don't want to exit if first platform has no devices of type
        // We do error checking next anyway, and can throw there if needed
#endif

        // Only squash CL_SUCCESS and CL_DEVICE_NOT_FOUND
        if (error != CL_SUCCESS && error != CL_DEVICE_NOT_FOUND) {
            cl::detail::errHandler(error, "clCreateContextFromType");
        }

        if (devices.size() > 0) {
            platform_id = (cl_context_properties)platforms[i]();
            break;
        }
    }
    if (platform_id == 0) {
        cl::detail::errHandler(CL_DEVICE_NOT_FOUND, "Not device found for the input type");
        return 0;
    }
    return platform_id;
}

cl::Context createCLGLContext(cl_device_type type) {
    cl::Platform platform;
    cl::Platform::get(&platform);


#if defined(__APPLE__) || defined(__MACOSX)
    // Apple (untested)
    cl_context_properties cps[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };

    cps[3] = getPlaformIdForType(type);

#else
#ifdef _WIN32
    // Windows
   cl_context_properties cps[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,
        (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };
#else
    // Linux
    cl_context_properties cps[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR,
        (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };
#endif
    cps[5] = getPlaformIdForType(type);
#endif

    return cl::Context(type, cps);
}

}

#endif
