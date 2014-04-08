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
#include <stack>
#include <stdexcept>

#if defined(WIN32) && !defined(__CYGWIN__)
	#include <io.h>
	#define WINBASE_DECLARE_GET_MODULE_HANDLE_EX
	#include <windows.h>
	#include <winbase.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <direct.h> // for _mkdir

	#define mkdir(x,y) _mkdir((x))
	#define stat64 _stati64
	#define access _access

	// set up for windows so acts just like unix access().
	#ifndef F_OK
		#define F_OK 4
	#endif
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
    #include <errno.h>
    #ifdef __APPLE__
        // To avoid deprected use of stat64
        #define stat64 stat
    #endif
#endif

#if !defined(S_ISDIR)
#  if defined( _S_IFDIR) && !defined( __S_IFDIR)
#    define __S_IFDIR _S_IFDIR
#  endif
#  define S_ISDIR(mode)    (mode&__S_IFDIR)
#endif

#include <mtUtils/FileUtils.h>

static const char * const PATH_SEPARATORS = "/\\";
	
std::string mtUtils::getFilePath(const std::string& fileName)
{
	std::string::size_type slash = fileName.find_last_of(PATH_SEPARATORS);
	if (slash==std::string::npos) return std::string();
	else return std::string(fileName, 0, slash);
}

void mtUtils::makeDirectory( const std::string &path )
{
	if (path.empty())
	{
		throw std::runtime_error("makeDirectory(): cannot create an empty directory");
	}

	struct stat64 stbuf;
	if( stat64( path.c_str(), &stbuf ) == 0 )
	{
		if( S_ISDIR(stbuf.st_mode))
			return;
		else
		{
			throw std::runtime_error("makeDirectory(): " + path + " already exists and is not a directory!");
		}
	}

	std::string dir = path;
	std::stack<std::string> paths;
	bool bTrue = true;
	while( bTrue )
	{
		if( dir.empty() )
			break;
		if( stat64( dir.c_str(), &stbuf ) < 0 )
		{
			switch( errno )
			{
				case ENOENT:
				case ENOTDIR:
					paths.push( dir );
					break;

				default:
					std::runtime_error("makeDirectory() errno:");
			}
		}
		dir = getFilePath(std::string(dir));
	}

	while( !paths.empty() )
	{
		std::string dir = paths.top();

		#if defined(WIN32)
			//catch drive name
			if (dir.size() == 2 && dir.c_str()[1] == ':') {
				paths.pop();
				continue;
			}
		#endif
		if( mkdir( dir.c_str(), 0755 )< 0 )
		{
			throw std::runtime_error("makeDirectory() : cannot create directory in this path " + dir);
		}
		paths.pop();
	}
	return;
}

void mtUtils::makeDirectoryForFile( const std::string &path )
{
	makeDirectory( getFilePath( path ));
}

bool mtUtils::fileExists(const std::string &filename)
{
	if( filename.empty())
	{
		return false;
	}
#if WIN32
	unsigned long dwAttrib = GetFileAttributesA(filename.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES); //&& !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
	if( access( filename.c_str(), F_OK ) == 0 )
	{
		return true;
	}
	return false;
#endif
}

