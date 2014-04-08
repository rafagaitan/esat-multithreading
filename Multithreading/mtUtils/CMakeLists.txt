SET(LIB_NAME mtUtils)
set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR})

SET(SOURCES
    FileUtils.cpp
    Barrier.cpp
)

SET(HEADERS
    ${HEADER_PATH}/Export.h
    ${HEADER_PATH}/ArgumentParser.h
    ${HEADER_PATH}/FileUtils.h
    ${HEADER_PATH}/ThreadPool.h
    ${HEADER_PATH}/ScopedThread.h
    ${HEADER_PATH}/ThreadSafeStack.h
    ${HEADER_PATH}/ThreadSafeQueue.h
    ${HEADER_PATH}/Barrier.h
    ${HEADER_PATH}/Algorithms.h
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

SET(LIBRARIES
)

SET(LIBRARIES_OPTIMIZED
)

SET(LIBRARIES_DEBUG
)

SETUP_CORELIB(${LIB_NAME} mt)
