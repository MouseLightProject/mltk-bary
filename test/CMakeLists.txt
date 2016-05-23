set(tgt mltk)

file(GLOB TEST_COMMON_SRCS common/*.h common/*.c)

foreach(testset 
        math 
        barycentricCPU
        barycentricGPU
        barycentricAVX
    )
    file(GLOB TESTS ${testset}/*.c)
    foreach(t ${TESTS})
      get_filename_component(exe ${t} NAME_WE)
      add_executable("${tgt}-${testset}-${exe}-test" ${t} ${TEST_COMMON_SRCS})
      set_property(TARGET "${tgt}-${testset}-${exe}-test" APPEND PROPERTY
                   INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/common)
      set_property(TARGET "${tgt}-${testset}-${exe}-test" PROPERTY
                   FOLDER engine/test/${testset})
      add_test("${tgt}-${testset}-${exe}" "${tgt}-${testset}-${exe}-test")
      target_add_engine("${tgt}-${testset}-${exe}-test")
    endforeach()
endforeach()

