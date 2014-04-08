cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)

SET(SOURCES
    session6.cpp
)

SET(HEADERS
    config.h
    CUDAUtils.h
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

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

cuda_add_executable(session6_cuda 
	${CUDA_SOURCES} 
	${CUDA_HEADERS}
	${SOURCES} 
	${HEADERS}
	OPTIONS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20}
)

SET_TARGET_PROPERTIES(session6_cuda PROPERTIES FOLDER "Applications")

FOREACH(BIN ${BIN_PATH})
	INSTALL(TARGETS session6_cuda
		RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
	)
ENDFOREACH()

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
    Matrix.h
)

source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("CUDA Header Files" FILES ${CUDA_HEADERS})

cuda_add_executable(session9_cuda 
	${CUDA_SOURCES} 
	${CUDA_HEADERS}
	${SOURCES} 
	${HEADERS}
	OPTIONS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20}
)

SET_TARGET_PROPERTIES(session9_cuda PROPERTIES FOLDER "Applications")

FOREACH(BIN ${BIN_PATH})
	INSTALL(TARGETS session9_cuda
		RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
	)
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
    config.h
    CUDAUtils.h
    OpenGLUtils.h
)

source_group("CUDA Source Files" FILES ${CUDA_SOURCES})
source_group("CUDA Header Files" FILES ${CUDA_HEADERS})

include_directories(
   ${OPENGL_INCLUDE_DIR}
   ${CMAKE_SOURCE_DIR}/OpenGL 
   ${CMAKE_SOURCE_DIR}/deps/i386-VC120/include
   ${GLFW_INCLUDE_DIR})

add_definitions(-DRESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
   
cuda_add_executable(session10_cuda 
	${CUDA_SOURCES} 
	${CUDA_HEADERS}
	${SOURCES} 
	${HEADERS}
	OPTIONS ${GENCODE_SM10} ${GENCODE_SM13} ${GENCODE_SM20}
)
set(LIBRARIES ${LIBRARIES} hdk ${GLEW_LIBRARIES})
target_link_libraries(session10_cuda ${LIBRARIES} hdk ${GLEW_LIBRARIES})

SET_TARGET_PROPERTIES(session10_cuda PROPERTIES FOLDER "Applications")

FOREACH(BIN ${BIN_PATH})
	INSTALL(TARGETS session10_cuda
		RUNTIME DESTINATION ${BIN} COMPONENT ${COMP}
	)
ENDFOREACH()