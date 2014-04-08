MAKE_DIRECTORY(${DEPLOY_DIR})

separate_arguments(DEPLOY_FILES)
foreach(f ${DEPLOY_FILES})
    MESSAGE("Copying file: "${f}" to: "${DEPLOY_DIR})
    configure_file(${f} ${DEPLOY_DIR} COPYONLY) #We use configure_file because it's the only way to follow the symlink
endforeach()
MESSAGE(${DEPLOY_DIR})

