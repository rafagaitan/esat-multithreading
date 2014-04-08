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
	SET(DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/i386-VC120)
    SET(CURL_DIR ${DEPS_DIR})
	SET(GLFW_DIR ${DEPS_DIR})
	SET(GLEW_DIR ${DEPS_DIR})
ELSEIF(APPLE)
	SET(DEPS_DIR ${CMAKE_SOURCE_DIR}/deps/macosx)
ENDIF()

SET(LIBLAS_DIR ${DEPS_DIR})

link_directories(${LIBRARY_OUTPUT_PATH})

#add_subdirectory(gtest-1.7.0)
add_subdirectory(Multithreading)
add_subdirectory(OpenGL)
add_subdirectory(CUDA)
add_subdirectory(OpenCL)

INCLUDE(InstallRequiredSystemLibraries)

IF(CMAKE_CPACK_COMMAND)
    OPTION(ENABLE_PACKAGING "Enable to activate packaging, it installs things in special way specially for macosx" OFF)
    IF(ENABLE_PACKAGING)
        INCLUDE(MacrosCPack)
    ENDIF()
ENDIF()

CONFIGURE_END()

