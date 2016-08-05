#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
include_directories(${PROJECT_SOURCE_DIR}/../tilebase)
include_directories(${PROJECT_SOURCE_DIR}/../tilebase/src)
include_directories(${ENGINE_DIR})

add_executable(scopeToVoxel scopeToVoxel.c ${PROJECT_SOURCE_DIR}/../tilebase/tilebase.h ${PROJECT_SOURCE_DIR}/src/resamplers.h)

#add_library(tilebase SHARED IMPORTED)

#set_target_properties(tilebase PROPERTIES
#    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
#    IMPORTED_LOCATION "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}tilebase${CMAKE_SHARED_LIBRARY_SUFFIX}"
#    )
#
#add_library(nd SHARED ${ND_LIBRARIES})

link_directories(${CMAKE_INSTALL_PREFIX}/build/mltk-bary)

find_library(TILEBASE_LIBRARY tilebase ${CMAKE_INSTALL_PREFIX}/tilebase)

#add_dependencies(scopeToVoxel engine)

target_add_engine(scopeToVoxel)


target_link_libraries(scopeToVoxel ${TILEBASE_LIBRARY})
#target_link_libraries(scopeToVoxel engine)
#message("CMAKE INSTALL PREFIX = ${CMAKE_INSTALL_PREFIX}")


nd_copy_plugins_to_target(scopeToVoxel ${plugins})

install(TARGETS scopeToVoxel DESTINATION ${CMAKE_INSTALL_PREFIX}/build/mltk-bary/bin)
