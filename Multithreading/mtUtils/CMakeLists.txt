SET(LIB_NAME mtUtils)
set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR})

SET(SOURCES
    FileUtils.cpp
    Barrier.cpp
)

SET(HEADERS
    ${HEADER_PATH}/Export.hpp
    ${HEADER_PATH}/ArgumentParser.hpp
    ${HEADER_PATH}/FileUtils.hpp
    ${HEADER_PATH}/ThreadPool.hpp
    ${HEADER_PATH}/ScopedThread.hpp
    ${HEADER_PATH}/ThreadSafeStack.hpp
    ${HEADER_PATH}/ThreadSafeQueue.hpp
    ${HEADER_PATH}/Barrier.hpp
    ${HEADER_PATH}/Algorithms.hpp
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
