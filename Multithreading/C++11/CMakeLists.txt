SET(FILES
	cpp11tests.cpp
	session2.cpp
	session3.cpp
	session4.cpp
	session5.cpp
)

FOREACH(F ${FILES})
    get_filename_component(APP_NAME ${F} NAME_WE)
    #set(APP_NAME ${F})

    set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR})

    set(SOURCES
	${F}
    )

    set(HEADERS
    )

    include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
    )

	SET(LIBRARIES mtUtils)

    SETUP_LAUNCHER(${APP_NAME} cpp11)

ENDFOREACH(F)
