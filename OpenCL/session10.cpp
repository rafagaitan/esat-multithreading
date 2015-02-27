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
#include <iostream>
#include <string>
#include <memory>
#include <chrono>

//gl includes
#include <GL/glew.h>

// hdk includes
#include <window.hpp>

#include "Config.hpp"

#include "cl.hpp"

#include "OpenCLUtils.hpp"
#include "OpenGLUtils.hpp"

const float ViewWidth = 800;
const float ViewHeight = 600;
const unsigned int meshWidth = 512;
const unsigned int meshHeight = 512;
const unsigned int NUM_VERTICES = meshWidth*meshHeight;

namespace math
{
    struct float2
    {
        float x,y;
    };

    struct float3
    {
        float x,y,z;
    };

    struct float4
    {
        float x,y,z,w;
    };
}


class SceneManager
{
private:

    // GL buffer data
    GLuint _destVertexVBO;
    GLuint _nDestVertices;
    // GL shader data
    GLuint _renderProgram;
    GLint  _vertexPositionCULocation;
    GLint _diffuseColorCULocation;
    //GLint _lightDirectionCULocation;
    GLint _lightColorCULocation;
    GLint _ambientColorCULocation;

    // CL data
    std::vector<cl::Memory> _destVBO_CL;

    // CL init data
    cl::Platform      _platform;
    cl::Context       _context;
    cl::Program       _program;
    cl::Kernel        _kernel_computeVertices;
    cl::CommandQueue  _queue;

    // scene manipulation
    float _zoom;
    float _rotateX;
    float _rotateY;
    float _translateX;
    float _translateY;

    // animation
    std::chrono::time_point<std::chrono::system_clock> _startTime;
public:
    SceneManager()
        :_destVertexVBO(0)
        ,_nDestVertices(0)
        ,_renderProgram(0)
        ,_vertexPositionCULocation(-1)
        ,_diffuseColorCULocation(-1)
        //,_lightDirectionCULocation(-1)
        ,_lightColorCULocation(-1)
        ,_ambientColorCULocation(-1)

        ,_destVBO_CL()

        ,_platform()
        ,_context()
        ,_program()
        ,_kernel_computeVertices()
        ,_queue()
        
        ,_zoom(5.0f)
        ,_rotateX(45.0f)
        ,_rotateY(45.0f)
        ,_translateX(0.0f)
        ,_translateY(0.0f)
    {

    }

    void OnInit(const Window& view)
    {
        int w,h;
        view.getWindowSize(w, h);
        initGL(w,h);
        try
        {
            initCL();
        }
        catch(cl::Error& err) 
        {
            std::cerr << "Error in OpenCL:" << err.what() << " Error code=" << err.err() << std::endl;

            exit(-1);
        }
        _startTime = std::chrono::system_clock::now();
    }

    void OnShutdown()
    {
        // cl uses c++ api so it will clean up resources automatically

        glDeleteBuffers(1,&_destVertexVBO);
        glDeleteProgram(_renderProgram);
    }

    void OnDraw(const Window& view)
    {
        float time = std::chrono::duration<float>(std::chrono::system_clock::now() - _startTime).count();
        computeVertices(time);
        int w,h;
        view.getWindowSize(w, h);
        glViewport(0,0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(30,(float)w/(float)h, 0.1, 1000.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glTranslatef(0,0,-_zoom);
        glTranslatef(_translateX,_translateY,0);
        glRotatef(_rotateX,1,0,0);
        glRotatef(_rotateY,0,1,0);	

        glUseProgram(_renderProgram);

        glEnableVertexAttribArray(_vertexPositionCULocation);
        glBindBuffer(GL_ARRAY_BUFFER, _destVertexVBO);
        glVertexAttribPointer(_vertexPositionCULocation, 4, GL_FLOAT, false, sizeof(math::float4), NULL);

        // state
        glPointSize(1.0f);
        glUniform4f(_diffuseColorCULocation, 0.75, 0.75, 0.75, 1);
        glUniform4f(_lightColorCULocation, 1, 1, 1, 1);
        glUniform4f(_ambientColorCULocation, 0, 0, 0, 1);

        glDrawArrays(GL_POINTS,0,_nDestVertices);

        glUseProgram(0);
    }

protected:
    void initCL()
    {
        _context = opencl::createCLGLContext(CL_DEVICE_TYPE_GPU);

        std::cerr << "After CLGL Context" << std::endl;

        std::vector<cl::Device> devices = _context.getInfo<CL_CONTEXT_DEVICES>();
        std::string sumIntSrc = opencl::loadKernel("kernels.cl");
        cl::Program::Sources source(1, std::make_pair(sumIntSrc.c_str(), sumIntSrc.size()));
        _program = cl::Program(_context, source);
        _program.build(devices);

        std::cerr << "After build program" << std::endl;

        _kernel_computeVertices = cl::Kernel(_program, "computeVertices");

        std::cerr << "After load kernel" << std::endl;

        _destVBO_CL.resize(1);
        _destVBO_CL[0] = cl::BufferGL(_context, CL_MEM_WRITE_ONLY, _destVertexVBO);

        std::cerr << "After BufferGL" << std::endl;

        _queue = cl::CommandQueue(_context, devices[0], 0);
        _kernel_computeVertices.setArg(0, _destVBO_CL[0]);
        _kernel_computeVertices.setArg(1, meshWidth);
        _kernel_computeVertices.setArg(2, meshHeight);
    }

    void initGL(int w, int h)
    {
        glClearColor(0.2f, 0.5f, 0.8f, 1.0f);
        glViewport(0,0, static_cast<GLsizei>(w), static_cast<GLsizei>(h)); // TODO catch resize and recompute this
        _renderProgram = createProgram();
        createVBO(&_destVertexVBO,NUM_VERTICES);
    }

GLuint createProgram() {
    GLuint program;
    GLuint vertexShader;
    GLuint fragmentShader;

    bool success = true;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    success &= opengl::CompileGLShaderFromFile(vertexShader, "Vertex.glsl");
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    success &= opengl::CompileGLShaderFromFile(fragmentShader, "Fragment.glsl");
    if(success) {
        program = glCreateProgram();

        glAttachShader(program, vertexShader); 
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);
        _vertexPositionCULocation = glGetAttribLocation( program, "vertexPosition");
        _diffuseColorCULocation = glGetUniformLocation(program, "diffuseColor");
        _lightColorCULocation = glGetUniformLocation(program, "lightColor");
        _ambientColorCULocation = glGetUniformLocation(program, "ambientColor");
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return program;
    }
    else {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }
}

    void createVBO(GLuint* vbo, unsigned int numVertices)
    {
        _nDestVertices = numVertices;
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(math::float4)*_nDestVertices, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void computeVertices(float simulationTime)
    {
        // set vbos
        _queue.enqueueAcquireGLObjects(&_destVBO_CL);
        // execute kernel
        _kernel_computeVertices.setArg(3,simulationTime);
        _queue.enqueueNDRangeKernel(
                _kernel_computeVertices, 
                cl::NullRange, 
                cl::NDRange(meshWidth,meshHeight), // global
                cl::NullRange);             // local
        // free vbos for gl rendering
        _queue.enqueueReleaseGLObjects(&_destVBO_CL);
    }


        
};

bool done = false;

int main()
{
    Window view;
    view.setWindowKeyCallback(
        [](int32 key, int32 /*scancode*/, int32 action, int32) 
    {
        if (action == 1) // released
        {
            switch (key)
            {
            case 256: //toggle lod level 't':
                done = true;
                break;
            }
        }
    });

    view.Init(800,600, "OpenCL-OpenGL Interop Sample");

    SceneManager scene;
    scene.OnInit(view);
    
    while (!view.ShouldClose() && ! done)
    {

        scene.OnDraw(view);
        //swap front and back buffers to show the rendered result
        view.Swap();

    }

    scene.OnShutdown();

    return EXIT_SUCCESS;
}

