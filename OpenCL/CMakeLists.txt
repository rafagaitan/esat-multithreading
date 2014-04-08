# packages                                                                                                                                                                                                 
find_package(OpenCL)
find_package(OpenGL)

SET(FILES
    session6.cpp
    session8.cpp
    session10.cpp
)

SET(HEADERS
    config.h
    OpenCLUtils.h
)

SET(OPENCL_SOURCES
    kernels.cl
)


FOREACH(F ${FILES})
    get_filename_component(APP_NAME ${F} NAME_WE)
    #set(APP_NAME ${F})

    set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR})

    set(SOURCES
        ${F}
    )

    include_directories(
       ${CMAKE_CURRENT_SOURCE_DIR}
       ${OPENCL_INCLUDE_DIR}
    )

    SET(LIBRARIES 
        ${OPENCL_LIBRARIES}
		${OPENGL_LIBRARIES}
    )

    if(APP_NAME MATCHES "session10")
		find_package(GLEW)
		
		set(LIBRARIES ${LIBRARIES} hdk ${GLEW_LIBRARIES})
	    set(SOURCES ${SOURCES} OpenGLUtils.cpp)
		set(HEADERS ${HEADERS} OpenGLUtils.h)
		include_directories(
		   ${OPENGL_INCLUDE_DIR}
	       ${CMAKE_SOURCE_DIR}/OpenGL 
		   ${CMAKE_SOURCE_DIR}/deps/i386-VC120/include
		   ${GLFW_INCLUDE_DIR})
    endif()

    source_group("OpenCL Source Files" FILES ${OPENCL_SOURCES})
    foreach(CL_FILE ${OPENCL_SOURCES})
        set_source_files_properties(${CL_FILE} PROPERTIES HEADER_FILE_ONLY TRUE)
        set(SOURCES ${SOURCES} ${CL_FILE})
    endforeach()
	ADD_DEFINITIONS(-DRESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    SETUP_LAUNCHER(${APP_NAME}_opencl opencl)

ENDFOREACH(F)

