cmake_minimum_required(VERSION 2.8)

IF(MSVC)
    SET(EXTRA_CXX_FLAGS "/Wall /W3 /EHsc")
    IF(BUILD_TREAT_WARNINGS_AS_ERRORS)
         SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} /WX")
    ENDIF()
    ADD_DEFINITIONS("-DWIN32")
ELSE()
    SET(EXTRA_CXX_FLAGS " -W -Wall -Wextra -Wparentheses -Wno-long-long -Wno-import -pedantic -Wreturn-type -Wmissing-braces -Wunknown-pragmas -Wunused")
    IF(BUILD_TREAT_WARNINGS_AS_ERRORS)
         SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Werror")
    ENDIF()
    IF(BUILD_ENABLE_EFFECTIVE_CPP_WARNINGS)
         SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Weffc++")
    ENDIF()
    IF(UNIX)
        IF(APPLE)
           IF(CMAKE_COMPILER_IS_CLANGXX OR (CMAKE_GENERATOR STREQUAL Xcode) )
                set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0x")
                set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
                SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS}  -std=c++0x -stdlib=libc++ -U__STRICT_ANSI__")
                SET(EXTRA_LINKER_FLAGS "${EXTRA_LINKER_FLAGS} -stdlib=libc++")
            ENDIF()
        ELSEIF(ANDROID)
            SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -std=gnu++0x -D__STDC_INT64__")
        ELSE()
            IF(CMAKE_COMPILER_IS_CLANGXX)
                SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -std=c++11")
            ELSE()
                SET(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -std=c++0x")
            ENDIF()
        ENDIF()
    ENDIF()
ENDIF()

IF(NOT CUDA_CONFIG_HAS_BEEN_RUN_BEFORE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}" CACHE STRING "Flags used by the compiler during all build types." FORCE)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINKER_FLAGS}" CACHE STRING "Flags used by the linker." FORCE)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXTRA_LINKER_FLAGS}" CACHE STRING "Flags used by the linker." FORCE)
    SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${EXTRA_LINKER_FLAGS}" CACHE STRING "Flags used by the linker." FORCE)
ENDIF()


SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

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

IF(NOT CUDA_CONFIG_HAS_BEEN_RUN_BEFORE)
    SET(CUDA_CONFIG_HAS_BEEN_RUN_BEFORE 1 CACHE INTERNAL "Flag to track whether this is the first time running CMake or if CMake has been configured before")
ENDIF(NOT CUDA_CONFIG_HAS_BEEN_RUN_BEFORE)
