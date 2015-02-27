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
//-------------------------------------------------------------------------------
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iterator>
#include <fstream>

#include "OpenGLUtils.hpp"

using namespace opengl;

bool opengl::printShaderInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten  = 0;
	char *infoLog;
	glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
	if (infologLength > 1)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("printShaderInfoLog: %s\n",infoLog);
		free(infoLog);
		return false;
	}
	else
	{
		printf("Shader Info Log: OK\n");
		return true;
	}
}

std::string opengl::loadShader(const char* name)
{
    std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

bool opengl::CompileGLShaderFromSource(GLuint Shader, const char* SourceCode)
{
	if(!SourceCode)
		return false;

	const char* ConstSourceCode = (const char*)SourceCode;
	glShaderSource(Shader, 1, &ConstSourceCode, NULL);
	
	glCompileShader(Shader);
	return printShaderInfoLog(Shader);
}

bool opengl::CompileGLShaderFromFile(GLuint Shader, const char* Filename)
{
	std::string SourceCode = loadShader(Filename);
    bool Success = CompileGLShaderFromSource(Shader, SourceCode.c_str());
	return Success;
}