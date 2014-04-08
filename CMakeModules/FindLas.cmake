# Author Rafa Gaitán <rgaitan@mirage-tech.com>

FIND_PATH(LIBLAS_INCLUDE_DIR liblas/liblas.hpp 
    ${LIBLAS_DIR}/include
    $ENV{LIBLAS_DIR}/include
    $ENV{LIBLAS_DIR}
    $ENV{LIBLASDIR}/include
    $ENV{LIBLASDIR}
    $ENV{LIBLAS_ROOT}/include
    NO_DEFAULT_PATH
)

FIND_PATH(LIBLAS_INCLUDE_DIR liblas/liblas.hpp)

MACRO(FIND_LIBLAS_LIBRARY MYLIBRARY MYLIBRARYNAME)

    FIND_LIBRARY("${MYLIBRARY}_DEBUG"
        NAMES "lib${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}.a" "${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}"
        PATHS
		${LIBLAS_DIR}/lib/Debug
        ${LIBLAS_DIR}/lib64/Debug
        ${LIBLAS_DIR}/lib
        ${LIBLAS_DIR}/lib64
        $ENV{LIBLAS_DIR}/lib/debug
        $ENV{LIBLAS_DIR}/lib64/debug
        $ENV{LIBLAS_DIR}/lib
        $ENV{LIBLAS_DIR}/lib64
        $ENV{LIBLAS_DIR}
        $ENV{LIBLASDIR}/lib
        $ENV{LIBLASDIR}/lib64
        $ENV{LIBLASDIR}
        $ENV{LIBLAS_ROOT}/lib
        $ENV{LIBLAS_ROOT}/lib64
        NO_DEFAULT_PATH
    )

    FIND_LIBRARY("${MYLIBRARY}_DEBUG"
        NAMES "lib${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}.a" "${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}"
        PATHS
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;LIBLAS_ROOT]/lib
        /usr/freeware/lib64
    )
    
    FIND_LIBRARY(${MYLIBRARY}
        NAMES "lib${MYLIBRARYNAME}.a" ${MYLIBRARYNAME}
        PATHS
	${LIBLAS_DIR}/lib/Release
        ${LIBLAS_DIR}/lib64/Release
        ${LIBLAS_DIR}/lib
        ${LIBLAS_DIR}/lib64
        $ENV{LIBLAS_DIR}/lib/Release
        $ENV{LIBLAS_DIR}/lib64/Release
        $ENV{LIBLAS_DIR}/lib
        $ENV{LIBLAS_DIR}/lib64
        $ENV{LIBLAS_DIR}
        $ENV{LIBLASDIR}/lib
        $ENV{LIBLASDIR}/lib64
        $ENV{LIBLASDIR}
        $ENV{LIBLAS_ROOT}/lib
        $ENV{LIBLAS_ROOT}/lib64
        NO_DEFAULT_PATH
    )

    FIND_LIBRARY(${MYLIBRARY}
        NAMES "lib${MYLIBRARYNAME}.a" ${MYLIBRARYNAME}
        PATHS
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;LIBLAS_ROOT]/lib
        /usr/freeware/lib64
    )
    
    IF( NOT ${MYLIBRARY}_DEBUG)
        SET(${MYLIBRARY}_DEBUG ${${MYLIBRARY}})
    ENDIF( NOT ${MYLIBRARY}_DEBUG)
           
ENDMACRO(FIND_LIBLAS_LIBRARY LIBRARY LIBRARYNAME)
IF(WIN32)
	FIND_LIBLAS_LIBRARY(LIBLAS_LIBRARY liblas)
	FIND_LIBLAS_LIBRARY(LIBLAS_C_LIBRARY liblas_c)
ELSE()
	FIND_LIBLAS_LIBRARY(LIBLAS_LIBRARY las)
	FIND_LIBLAS_LIBRARY(LIBLAS_C_LIBRARY las_c)
ENDIF()

SET(LIBLAS_LIBRARIES
	${LIBLAS_LIBRARY}
	${LIBLAS_C_LIBRARY}
)

SET(LIBLAS_LIBRARIES_DEBUG
	${LIBLAS_LIBRARY_DEBUG}
	${LIBLAS_C_LIBRARY_DEBUG}
)

# handle the QUIETLY and REQUIRED arguments and set
# LIBLAS_FOUND to TRUE as appropriate
INCLUDE( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( LIBLAS DEFAULT_MSG LIBLAS_INCLUDE_DIR LIBLAS_LIBRARIES)

