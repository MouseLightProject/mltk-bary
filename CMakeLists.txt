cmake_minimum_required(VERSION 2.8)
project(mltk-barry)

SET( CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-fPIC -mavx2 -mfma" )
SET( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fPIC -mavx2 -mfma" )

include(cmake/show.cmake)
include(cmake/ide.cmake)
include(cmake/rpath.cmake)

#find_package(ND)
#include_directories(${ND_INCLUDE_DIRS})
#;-Xptxas="-v"

file(GLOB_RECURSE SRCS src/*.c src/*.h)
include_directories(src)

#find_package(CUDA 5.0)
#file(GLOB_RECURSE CUSRCS src/*.cu)
#cuda_compile(CUFILES ${CUSRCS} STATIC
#      OPTIONS
#      DEBUG -g;-G;-lineinfo
#      #DEBUG -g;-G;-lineinfo;-Xptxas="-v"
#      RELEASE --use_fast_math
#      RELWITHDEBINFO --use_fast_math;-g;-G;-lineinfo
#      MINSIZEREL --use_fast_math
#)
#list(APPEND SRCS ${CUFILES})

add_library(engine SHARED ${SRCS})
target_link_libraries(engine rt)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
function(target_add_engine tgt)
    set_property(TARGET ${tgt} APPEND PROPERTY 
                 INCLUDE_DIRECTORIES ${ENGINE_DIR})
    target_link_libraries(${tgt} engine)
    #nd_add_to_target(${tgt} ndio-tiff)
endfunction()

include(CTest)
#enable_testing()
#add_subdirectory(test)
#add_subdirectory(app/scopeToVoxel)
