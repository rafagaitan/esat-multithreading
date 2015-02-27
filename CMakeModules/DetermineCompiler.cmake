# - If Visual Studio is being used, this script sets the variable SYSTEM_COMPILER
# The principal reason for this is due to MSVC 8.0 SP0 vs SP1 builds.
#
# Variable:
#   SYSTEM_COMPILER
#   SYSTEM_ARCH
#   SYSTEM_NAME
#   SYSTEM_ID
# 

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_COMPILER_IS_CLANGXX 1)
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
   SET(SYSTEM_COMPILER  "GCC")
ELSEIF(MSVC60)
    SET(SYSTEM_COMPILER "VC60")
ELSEIF(MSVC70)
    SET(SYSTEM_COMPILER "VC70")
ELSEIF(MSVC71)
    SET(SYSTEM_COMPILER "VC71")
ELSEIF(MSVC80)
    SET(SYSTEM_COMPILER "VC80")
ELSEIF(MSVC90)
    SET(SYSTEM_COMPILER "VC90")
ELSEIF(MSVC10)
    SET(SYSTEM_COMPILER "VC100")
ELSEIF(MSVC11)
    SET(SYSTEM_COMPILER "VC110")
ELSEIF(MSVC12)
    SET(SYSTEM_COMPILER "VC120")
ENDIF()


IF(MSVC80)
    MESSAGE(STATUS "Checking if compiler has service pack 1 installed...")
    FILE(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx" "int main() {return 0;}\n")

    TRY_COMPILE(_TRY_RESULT
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
        CMAKE_FLAGS -D CMAKE_VERBOSE_MAKEFILE=ON
        OUTPUT_VARIABLE OUTPUT
        )

    IF(_TRY_RESULT)
        # parse for exact compiler version
        STRING(REGEX MATCH "Compiler Version [0-9]+.[0-9]+.[0-9]+.[0-9]+" vc_compiler_version "${OUTPUT}")
        IF(vc_compiler_version)
            #MESSAGE("${vc_compiler_version}")
            STRING(REGEX MATCHALL "[0-9]+" CL_VERSION_LIST "${vc_compiler_version}")
            LIST(GET CL_VERSION_LIST 0 CL_MAJOR_VERSION)
            LIST(GET CL_VERSION_LIST 1 CL_MINOR_VERSION)
            LIST(GET CL_VERSION_LIST 2 CL_PATCH_VERSION)
            LIST(GET CL_VERSION_LIST 3 CL_EXTRA_VERSION)
        ENDIF(vc_compiler_version)

        # Standard vc80 is 14.00.50727.42, sp1 14.00.50727.762, sp2?
        # Standard vc90 is 9.0.30729.1, sp1 ?
        IF(CL_EXTRA_VERSION EQUAL 762)
            SET(SYSTEM_COMPILER "VC80sp1")
        ELSE(CL_EXTRA_VERSION EQUAL 762)
            SET(SYSTEM_COMPILER "VC80")
        ENDIF(CL_EXTRA_VERSION EQUAL 762)

        # parse for exact visual studio version
        #IF(MSVC_IDE)
        # string(REGEX MATCH "Visual Studio Version [0-9]+.[0-9]+.[0-9]+.[0-9]+" vs_version "${OUTPUT}")
        # IF(vs_version)
        # MESSAGE("${vs_version}")
        # string(REGEX MATCHALL "[0-9]+" VS_VERSION_LIST "${vs_version}")
        # list(GET VS_VERSION_LIST 0 VS_MAJOR_VERSION)
        # list(GET VS_VERSION_LIST 1 VS_MINOR_VERSION)
        # list(GET VS_VERSION_LIST 2 VS_PATCH_VERSION)
        # list(GET VS_VERSION_LIST 3 VS_EXTRA_VERSION)
        # ENDIF(vs_version)
        #ENDIF(MSVC_IDE)
    ENDIF(_TRY_RESULT)
ENDIF(MSVC80)

IF(APPLE)
    IF(CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET(SYSTEM_ARCH "x86_64")
    else()
        SET(SYSTEM_ARCH ${CMAKE_SYSTEM_PROCESSOR})
    endif()
ELSE()
# resolve architecture. The reason i "change" i686 to i386 is that debian packages
# require i386 so this is for the future
    IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "i686")
        SET(SYSTEM_ARCH "i386")
    ELSE()
        If(MSVC)
            if(CMAKE_CL_64)
                SET(SYSTEM_ARCH "x64")
            else()
                SET(SYSTEM_ARCH ${CMAKE_SYSTEM_PROCESSOR})
            endif()
        else()
            SET(SYSTEM_ARCH ${CMAKE_SYSTEM_PROCESSOR})
        endif()
    ENDIF()
ENDIF()

# set a default system name - use CMake setting (Linux|Windows|...)
SET(SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
#detect codename for linux systems
if(UNIX AND NOT APPLE)
    execute_process(COMMAND /usr/bin/lsb_release -c -s 
	            OUTPUT_VARIABLE SYSTEM_CODENAME
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND /usr/bin/lsb_release -i -s 
	            OUTPUT_VARIABLE SYSTEM_OS_ID
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND /usr/bin/lsb_release -r -s 
	            OUTPUT_VARIABLE SYSTEM_OS_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    SET(SYSTEM_ID   "${SYSTEM_CODENAME}-${SYSTEM_ARCH}-${SYSTEM_COMPILER}")
else()
    SET(SYSTEM_ID   "${SYSTEM_ARCH}-${SYSTEM_COMPILER}")
endif()

