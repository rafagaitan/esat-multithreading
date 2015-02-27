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


#define GLFW_INCLUDE_GLU

#include <iostream>
#include <functional>

#include <GL/glew.h>
#include <glfw3.h>


#include "window.hpp"
#include "types.hpp"


struct GLFWInitializer
{
    GLFWInitializer()
    {
        //Iniciamos glfw
        if (glfwInit() == GL_FALSE){
            std::cout << "Error initializing GLFW" << std::endl;
        }else{
            std::cout << "GLFW initialized correctly" << std::endl;
        }

        //Una vez abierta, inicializamos glew
        //if (glewInit() != GLEW_OK){
        //    std::cout << "Failed to initialize GLEW" << std::endl;
        //}else{
        //    std::cout << "GLEW initialized correctly" << std::endl;
        //}
    }
};

struct GLEWinitializer
{
    GLEWinitializer()
    {
        //Una vez abierta, inicializamos glew
        if (glewInit() != GLEW_OK){
            std::cout << "Failed to initialize GLEW" << std::endl;
        }else{
            std::cout << "GLEW initialized correctly" << std::endl;
        }
    }
};

Window::Window():
    windowHandler(0),
    keyCallback(0)
{
    static GLFWInitializer s_glfwInit;
}

bool Window::Init(int32 width, int32 height,const char* title, ubyte /*screenOpcs*/)
{
    if(windowHandler)
        return true;

    windowHandler = glfwCreateWindow(width, height, title, NULL, NULL);

    //If the OpenWindow function returns GL_FALSE we will show an eerror and return -1
    if (!windowHandler){

        std::cout<<"Error opening window\n"<<std::endl;
        return false;

        //If the window opens OK we return 0
    }

    // set up user pointer to the GLFWindow
    glfwSetWindowUserPointer(windowHandler, this );

    // now set up the key handler (it will be the same for all the windows but each one will get the user pointer)
    glfwSetKeyCallback(windowHandler, &Window::setGLFWKeyCallback);

    //Show OPENGL info
    std::cout << "Opengl max supported version: " << glfwGetWindowAttrib(windowHandler, GLFW_CONTEXT_VERSION_MAJOR) << std::endl;
    //std::cout << "Compatible video modes: " << glfwGetVideoModes(); << endl
    /* Make the window's context current */
    glfwMakeContextCurrent(windowHandler);

    // inicializamos glew
    static GLEWinitializer s_glewInit;

    return true;
}

Window::~Window(){
    glfwDestroyWindow(windowHandler);
}

//!Processes all the events since the last call of this function.. IT IS CALLED AUTOMATICALLY WHEN WE CALL THE swap() FUNCTION.
/*void Window::ProcessEvents(){
glfwPollEvents();
}*/

//!Swaps the buffers
/*!Swaps the back buffer for the front buffer.*/
void Window::Swap(){
    glfwPollEvents();
    glfwSwapBuffers(windowHandler);
}

//!Sets the clear color.*/
/*!Sets the new RGBA color that will be used to clear the screen.*/
void Window::SetClearColor(glm::vec4& /*newClearColor*/){

}

int32 Window::ShouldClose(){
    return glfwWindowShouldClose(windowHandler);
}

//!This functions enables the vertical synchronization.
/*!This function synchronizes every swap of the buffers with the screen refresh. It limits the game fps to the screen refresh rate*/
void Window::EnableVsync(){
    glfwSwapInterval(1);
}

//!This functions disables the vertical synchronization.
/*!This function does't set a fps limit but can cause tearing*/
void Window::DisableVsync(){
    glfwSwapInterval(0);
}

//!Ends the application and closes the context.
/*!It will free the momory reserved for the window context and close the program.*/
void Window::Finish(){
    glfwDestroyWindow(windowHandler);
}

void Window::setWindowKeyCallback(CBFunctionKey* cbFunction){

    if (cbFunction != NULL){
        keyCallback = cbFunction;
    }

}

void Window::setGLFWKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){

    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (w->keyCallback)
        w->keyCallback(key,scancode,action,mods);
}
