cmake_minimum_required(VERSION 2.8)

IF(APPLE)
    SET(EXTRA_CXX_FLAGS "-stdlib=libstdc++")
    SET(EXTRA_LINKER_FLAGS "-stdlib=libstdc++")

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}" CACHE STRING "Flags used by the compiler during all build types." FORCE)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINKER_FLAGS}" CACHE STRING "Flags used by the linker." FORCE)
ENDIF()

# packages                                                                                                                                                                                                 



#session 7
SET(SOURCES
    session7.cpp
)

SET(HEADERS
    Config.hpp
    CUDAUtils.hpp
)

SET(CUDA_SOURCES
    SumIntVectors.cu
)

SET(CUDA_HEADERS
    SumIntVectors.cuh
)

include_directories(
    ${CMAKE_SOURCE_DIR}/Multithreading
)

source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("CUDA Header Files" FILES ${CUDA_HEADERS})

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

cuda_add_executable(session7_cuda 
    ${CUDA_SOURCES} 
    ${CUDA_HEADERS}
    ${SOURCES} 
    ${HEADERS}
    OPTIONS  ${GENCODE_SM20}
)

SET_TARGET_PROPERTIES(session7_cuda PROPERTIES FOLDER "Applications")

FOREACH(BIN ${BIN_PATH})
    INSTALL(TARGETS session7_cuda
        RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
    )
ENDFOREACH()

#session9
SET(SOURCES
    session9.cpp
)


SET(CUDA_SOURCES
    MatrixMult.cu
)

SET(CUDA_HEADERS
    MatrixMult.cuh
)

SET(HEADERS
    ${HEADERS}
    Matrix.hpp
    Multithreading/mtUtils/Algorithms.hpp
)

source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("CUDA Header Files" FILES ${CUDA_HEADERS})

cuda_add_executable(session9_cuda 
    ${CUDA_SOURCES} 
    ${CUDA_HEADERS}
    ${SOURCES} 
    OPTIONS ${GENCODE_SM20}
)

FIND_LIBRARY(CUDA_CUDA_LIBRARY
    cuda
    PATHS 
    ${CUDA_INSTALL_PREFIX}/lib 
    $ENV{CUDA_PATH}/lib/Win32
    /Developer/NVIDIA/CUDA-6.5/lib/
    DOC "\"cuda\" library"
)

target_link_libraries(session9_cuda ${CUDA_CUDA_LIBRARY})

SET_TARGET_PROPERTIES(session9_cuda PROPERTIES FOLDER "Applications")

if(WIN32)
    add_custom_command(TARGET session9_cuda 
        PRE_BUILD 
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build-cubin.bat ${CMAKE_BINARY_DIR}/CUDA 
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

FOREACH(BIN ${BIN_PATH})
    INSTALL(TARGETS session9_cuda
        RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
    )
    IF(WIN32)
        INSTALL(FILES ${CMAKE_BINARY_DIR}/CUDA/MatrixMult.cubin
            DESTINATION  ${BIN} COMPONENT ${COMP}
        )
    ENDIF()
ENDFOREACH()

# session 10
find_package(OpenGL)
find_package(GLEW)


SET(SOURCES
    session10.cpp
    OpenGLUtils.cpp
)


SET(CUDA_SOURCES
    ComputeVertices.cu
)

SET(CUDA_HEADERS
    ComputeVertices.cuh
)

SET(HEADERS
    Config.hpp
    CUDAUtils.hpp
    OpenGLUtils.hpp
)

source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("CUDA Header Files" FILES ${CUDA_HEADERS})

include_directories(
   ${OPENGL_INCLUDE_DIR}
   ${CMAKE_SOURCE_DIR}/OpenGL
   ${CMAKE_SOURCE_DIR}/deps/common/include 
   ${GLFW_INCLUDE_DIR})

add_definitions(-DRESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
   
cuda_add_executable(session10_cuda 
    ${CUDA_SOURCES} 
    ${CUDA_HEADERS}
    ${SOURCES} 
    ${HEADERS}
    OPTIONS ${GENCODE_SM13} ${GENCODE_SM20}
)
set(LIBRARIES ${LIBRARIES} hdk ${GLEW_LIBRARIES})
target_link_libraries(session10_cuda ${LIBRARIES} hdk ${GLEW_LIBRARIES})

SET_TARGET_PROPERTIES(session10_cuda PROPERTIES FOLDER "Applications")

FOREACH(BIN ${BIN_PATH})
    INSTALL(TARGETS session10_cuda
        RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
    )
ENDFOREACH()
