/*
Author: Ismael Castellanos Ruiz
Data: 11/01/2014
Contact: iscaru1988@gmail.com
*/

#ifndef __TYPES_H__
#define __TYPES_H__ 1

namespace hdk
{
    typedef char            char8;
    typedef char            int8;
    typedef unsigned char   ubyte;
    typedef short           int16;
    typedef unsigned short  uint16;
    typedef int             int32;
    typedef unsigned int    uint32;
    //#define uint64  (long long)
    typedef float           float32;
    typedef double          float64;

    //Typedef for callback functions
    typedef void (CBFunctionKey)(int32, int32, int32, int32);
    typedef void (CBFunctionMouseMove)(int32, int32);
    typedef void (CBFunctionMouseButton)(int32, int32);
    typedef void (CBFunctionMouseWheel)(int32);

    enum ShaderTypes{
      HDK_VERTEX_SHADER,
      HDK_FRAGMENT_SHADER,
      HDK_GEOMETRY_SHADER
    };

    enum DataTypes{
      HDK_INT,
      HDK_FLOAT,
      HDK_FLOAT_VEC_2,
      HDK_FLOAT_VEC_3,
      HDK_FLOAT_VEC_4,
      HDK_FLOAT_MAT_2,
      HDK_FLOAT_MAT_3,
      HDK_FLOAT_MAT_4
    };

    enum BufferTargets{
      HDK_ARRAY_BUFFER,
      HDK_ELEMENT_ARRAY_BUFFER
    };

}

#endif
