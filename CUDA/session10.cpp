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

#include <cuda_gl_interop.h>

// hdk includes
#include <window.hpp>

#include "CUDAUtils.hpp"
#include "OpenGLUtils.hpp"
#include "ComputeVertices.cuh"

const float ViewWidth = 800;
const float ViewHeight = 600;

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
    GLint _lightDirectionCULocation;
    GLint _lightColorCULocation;
    GLint _ambientColorCULocation;

    // CUDA data
    cudaGraphicsResource* _destVBO_CUDA;

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
        ,_lightDirectionCULocation(-1)
        ,_lightColorCULocation(-1)
        ,_ambientColorCULocation(-1)

        ,_destVBO_CUDA(0)
        
        ,_zoom(5.0f)
        ,_rotateX(45.0f)
        ,_rotateY(45.0f)
        ,_translateX(0.0f)
        ,_translateY(0.0f)
    {

    }

    void OnInit()
    {
        initGL();
        initCUDA();
        
        _startTime = std::chrono::system_clock::now();
    }

    void OnShutdown()
    {
        cudaGraphicsUnregisterResource(_destVBO_CUDA);
        glDeleteBuffers(1,&_destVertexVBO);
        glDeleteProgram(_renderProgram);
    }

    void OnDraw()
    {
        float time = std::chrono::duration<float>(std::chrono::system_clock::now() - _startTime).count();
        computeVertices(time);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(30,(float)ViewWidth/ViewHeight, 0.1, 1000.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glTranslatef(0,0,-_zoom);
        glTranslatef(_translateX,_translateY,0);
        glRotatef(_rotateX,1,0,0);
        glRotatef(_rotateY,0,1,0);	

        glUseProgram(_renderProgram);

        glEnableVertexAttribArray(_vertexPositionCULocation);
        glBindBuffer(GL_ARRAY_BUFFER, _destVertexVBO);
        glVertexAttribPointer(_vertexPositionCULocation, 4, GL_FLOAT, false, sizeof(float4), NULL);

        // state
        glPointSize(1.0f);
        glUniform4f(_diffuseColorCULocation, 0.75, 0.75, 0.75, 1);
        glUniform4f(_lightColorCULocation, 1, 1, 1, 1);
        glUniform4f(_ambientColorCULocation, 0, 0, 0, 1);

        glDrawArrays(GL_POINTS,0,_nDestVertices);

        glUseProgram(NULL);
    }

protected:
    void initCUDA()
    {
        cudaGraphicsGLRegisterBuffer(&_destVBO_CUDA, _destVertexVBO, cudaGraphicsMapFlagsWriteDiscard);
    }

    void initGL()
    {
        glClearColor(0.2f, 0.5f, 0.8f, 1.0f);

        glViewport(0,0, static_cast<GLsizei>(ViewWidth), static_cast<GLsizei>(ViewHeight)); // TODO catch resize and recompute this

        _renderProgram = createProgram();
        createVBO(&_destVertexVBO,NUM_VERTICES);
    }

    GLuint createProgram()
    {
        GLuint program;
        GLuint vertexShader;
        GLuint fragmentShader;

        bool success = true;

        vertexShader = glCreateShader(GL_VERTEX_SHADER);
        success &= opengl::CompileGLShaderFromFile(vertexShader, (std::string(RESOURCES_PATH) + std::string("/Vertex.glsl")).c_str());

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        success &= opengl::CompileGLShaderFromFile(fragmentShader, (std::string(RESOURCES_PATH) + std::string("/Fragment.glsl")).c_str());

        if(success)
        {
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
        else
        {
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
        glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*_nDestVertices, NULL, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void computeVertices(float simulationTime)
    {
        try
        {
            size_t num_bytes;
            float4* vertexPositions; 
            cudaGraphicsMapResources(1, &_destVBO_CUDA, 0);  
            cudaGraphicsResourceGetMappedPointer((void**)&vertexPositions, &num_bytes, _destVBO_CUDA);

            ::computeVertices(vertexPositions, meshWidth, meshHeight, simulationTime);

            cudaGraphicsUnmapResources(1, &_destVBO_CUDA, 0);
        }
        catch(cuda::cuda_exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
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

    view.Init(800,600, "CUDA-OpenGL Interop Sample");

    SceneManager scene;
    scene.OnInit();
    
    while (!view.ShouldClose() && ! done)
    {

        scene.OnDraw();
        //swap front and back buffers to show the rendered result
        view.Swap();

    }

    scene.OnShutdown();

    return EXIT_SUCCESS;
}

