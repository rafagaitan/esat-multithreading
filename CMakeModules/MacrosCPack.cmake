# This script sets up packaging targets for each "COMPONENT" as specified in INSTALL commands
#
# for each component a CPackConfig-<component>.cmake is generated in the build tree
# and a target is added to call cpack for it (e.g. package_openscenegaph
# A target for generating a package with everything that gets INSTALLED is generated (package_openscenegraph-all)
# A target for making all of the abaove packages is generated (package_ALL)
#
# package filenames are created on the form <package>-<platform>-<arch>[-<compiler>]-<build_type>[-static].tar.gz
# ...where compiler optionally set using a cmake gui (PACKAGE_CPACK_COMPILER). This script tries to guess compiler version for msvc generators
# ...build_type matches CMAKE_BUILD_TYPE for all generators but the msvc ones

# resolve architecture. The reason i "change" i686 to i386 is that debian packages
# require i386 so this is for the future
IF("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "i686")
    SET(SYSTEM_ARCH "i386")
ELSE()
    SET(SYSTEM_ARCH ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

# set a default system name - use CMake setting (Linux|Windows|...)
SET(SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
#message(STATUS "CMAKE_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}")
#message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

# for msvc the SYSTEM_NAME is set win32/64 instead of "Windows"
IF(MSVC)
    IF(CMAKE_CL_64)
        SET(SYSTEM_NAME "win64")
    ELSE()
        SET(SYSTEM_NAME "win32")
    ENDIF()
ENDIF()

# Guess the compiler (is this desired for other platforms than windows?)
INCLUDE(DetermineCompiler)

IF(PACKAGE_CPACK_COMPILER)
  SET(PACKAGE_CPACK_SYSTEM_SPEC_STRING ${SYSTEM_NAME}-${SYSTEM_ARCH}-${PACKAGE_CPACK_COMPILER})
ELSE()
  SET(PACKAGE_CPACK_SYSTEM_SPEC_STRING ${SYSTEM_NAME}-${SYSTEM_ARCH})
ENDIF()


## variables that apply to all packages
#(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${VERSION}")

# these goes for all platforms. Setting these stops the CPack.cmake script from generating options about other package compression formats (.z .tz, etc.)
IF(WIN32)
    SET(CPACK_GENERATOR "NSIS;ZIP" CACHE STRING "CPack package generator type (i.e ZIP,NSIS,TGZ,DEB,RPM, -- see CPack for valid stypes")
ELSE()
    IF(APPLE)
        SET(CPACK_BINARY_DRAGNDROP ON)
        SET(CPACK_GENERATOR "DragNDrop" CACHE STRING "CPack package generator type (i.e ZIP,NSIS,TGZ,DEB,RPM, -- see CPack for valid stypes")
    ELSE()
        SET(CPACK_GENERATOR "DEB;TGZ" CACHE STRING "CPack package generator type (i.e ZIP,NSIS,TGZ,DEB,RPM, -- see CPack for valid stypes")
        SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${OS_ARCH})
        SET(CPACK_DEBIAN_PACKAGE_SECTION "misc")
        SET(CPACK_DEBIAN_PACKAGE_DEPENDS "")
	#"libqtgui4 (>= 4.4.0),
        #libqtcore4 (>= 4.4.0), libqt4-xml (>= 4.4.0), libavcodec-extra-52 (>= 0.5),
        #libavformat-extra-52 (>= 0.5), libavdevice-extra-52 (>= 0.5), libswscale-extra-0 (>= 0.5),
        #libavfilter-extra-0 (>= 0.5), libavutil-extra-49 (>= 0.5)")
    ENDIF()
ENDIF()
SET(CPACK_SOURCE_GENERATOR "TGZ")


# for ms visual studio we use it's internally defined variable to get the configuration (debug,release, ...) 
IF(MSVC_IDE)
    SET(PACKAGE_TARGET_PREFIX "Package ")
ELSE()
    SET(PACKAGE_TARGET_PREFIX "package_")
ENDIF()
IF(CMAKE_BUILD_TYPE)
	SET(PACKAGE_CPACK_CONFIGURATION ${CMAKE_BUILD_TYPE})
ELSE()
	SET(PACKAGE_CPACK_CONFIGURATION "Release")
ENDIF()

# Get all defined components
#GET_CMAKE_PROPERTY(CPACK_COMPONENTS_ALL COMPONENTS)

# Create a target that will be used to generate all packages defined below
SET(PACKAGE_ALL_TARGETNAME "${PACKAGE_TARGET_PREFIX}ALL")
ADD_CUSTOM_TARGET(${PACKAGE_ALL_TARGETNAME})

MACRO(GENERATE_PACKAGING_TARGET package_name)
    SET(CPACK_PACKAGE_NAME ${package_name})

    # the doc packages don't need a system-arch specification
    IF(${package} MATCHES -doc)
        SET(PACKAGE_PACKAGE_FILE_NAME ${package_name}-${VERSION})
    ELSE()
        SET(PACKAGE_PACKAGE_FILE_NAME ${package_name}-${VERSION}-${PACKAGE_CPACK_SYSTEM_SPEC_STRING})
    ENDIF()

    SET(PACKAGE_PACKAGE_EXECUTABLE "${package_name}")

    CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/CMakeModules/MacrosCPackConfig.cmake.in" "${CMAKE_BINARY_DIR}/CPackConfig-${package_name}.cmake" IMMEDIATE)

    SET(PACKAGE_TARGETNAME "${PACKAGE_TARGET_PREFIX}${package_name}")

    # Create a target that creates the current package
    # and rename the package to give it proper filename
    ADD_CUSTOM_TARGET(${PACKAGE_TARGETNAME})
    ADD_CUSTOM_COMMAND(TARGET ${PACKAGE_TARGETNAME}
        COMMAND ${CMAKE_CPACK_COMMAND} -C ${PACKAGE_CPACK_CONFIGURATION} --config ${CMAKE_BINARY_DIR}/CPackConfig-${package_name}.cmake
        COMMENT "Run CPack packaging for ${package_name}..."
    )
    # Add the exact same custom command to the all package generating target. 
    # I can't use add_dependencies to do this because it would allow parallell building of packages so am going brute here
    ADD_CUSTOM_COMMAND(TARGET ${PACKAGE_ALL_TARGETNAME}
        COMMAND ${CMAKE_CPACK_COMMAND} -C ${PACKAGE_CPACK_CONFIGURATION} --config ${CMAKE_BINARY_DIR}/CPackConfig-${package_name}.cmake
    )
ENDMACRO(GENERATE_PACKAGING_TARGET)

# Create configs and targets for a package including all components
SET(PACKAGE_CPACK_COMPONENT ALL)
GENERATE_PACKAGING_TARGET(${SHORT_NAME}-all)

