# packages                                                                                                                                                                                                 

SET(FILES
    session8.cpp
    session10.cpp
)

SET(HEADERS
    Config.hpp
    OpenCLUtils.hpp
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
       ${CMAKE_SOURCE_DIR}/Multithreading
       ${CMAKE_CURRENT_SOURCE_DIR}
       ${OPENCL_INCLUDE_DIR}
    )

    SET(LIBRARIES 
        ${OPENCL_LIBRARIES}
        ${OPENGL_LIBRARIES}
        mtUtils
    )
    if(APP_NAME MATCHES "session10")
        find_package(GLEW)
        set(LIBRARIES ${LIBRARIES} hdk ${GLEW_LIBRARIES})
        set(SOURCES ${SOURCES} OpenGLUtils.cpp)
        set(HEADERS ${HEADERS} OpenGLUtils.hpp)
        include_directories(
            ${OPENGL_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/OpenGL 
            ${CMAKE_SOURCE_DIR}/deps/common/include
            ${GLFW_INCLUDE_DIR})
        FOREACH(BIN ${BIN_PATH})
            INSTALL(FILES Fragment.glsl Vertex.glsl
                DESTINATION ${BIN} COMPONENT ${COMP}
            )
        ENDFOREACH()
    endif()

    source_group("OpenCL Source Files" FILES ${OPENCL_SOURCES})
    foreach(CL_FILE ${OPENCL_SOURCES})
        set_source_files_properties(${CL_FILE} PROPERTIES HEADER_FILE_ONLY TRUE)
        set(SOURCES ${SOURCES} ${CL_FILE})
    endforeach()
    ADD_DEFINITIONS(-DRESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    SETUP_LAUNCHER(${APP_NAME}_opencl opencl)

ENDFOREACH(F)

FOREACH(BIN ${BIN_PATH})
    INSTALL(FILES ${OPENCL_SOURCES}
        DESTINATION ${BIN} COMPONENT ${COMP}
    )
ENDFOREACH()

