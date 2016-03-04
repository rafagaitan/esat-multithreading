project(ESAT)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

IF(COMMAND CMAKE_POLICY)
    cmake_policy(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

SET(MAJOR_VERSION 1)
SET(MINOR_VERSION 0)
SET(PATCH_VERSION 0)
SET(VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

# This values are used in macros to generate correct names and 
# correct paths to install stuff
SET(SHORT_NAME "esat")
SET(PLUGIN_PREFIX "${SHORT_NAME}_")

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")

INCLUDE(Macros)

CONFIGURE_DEFAULTS()

IF(WIN32)
    IF(CMAKE_CL_64)
        SET(DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/x64-VC120)
    ELSE()
        SET(DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/i386-VC120)
    ENDIF()
    SET(GLFW_DIR ${DEPS_DIR})
    SET(GLEW_DIR ${DEPS_DIR})
ELSEIF(APPLE)
    SET(DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/macosx)
ENDIF()


link_directories(${LIBRARY_OUTPUT_PATH})

add_subdirectory(Multithreading)

find_package(OpenGL)

add_subdirectory(OpenGL)


find_package(CUDA)
if(WIN32) # temporal until I figure out how to fix building
  if(CUDA_FOUND)
      add_subdirectory(CUDA)
  endif()
endif()

find_package(OpenCL)
if(OpenCL_FOUND)
    add_subdirectory(OpenCL)
endif()

INCLUDE(InstallRequiredSystemLibraries)

IF(CMAKE_CPACK_COMMAND)
    OPTION(ENABLE_PACKAGING "Enable to activate packaging, it installs things in special way specially for macosx" OFF)
    IF(ENABLE_PACKAGING)
        INCLUDE(MacrosCPack)
    ENDIF()
ENDIF()

CONFIGURE_END()

