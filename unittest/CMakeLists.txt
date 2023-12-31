# cmake version
cmake_minimum_required(VERSION 3.16.3)

# project info
project(pos_unittest LANGUAGES CXX CUDA)

# set executable output path
set(PATH_EXECUTABLE bin)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_EXECUTABLE})
SET(EXECUTABLE_OUTPUT_PATH ../${PATH_EXECUTABLE})

# find CUDA Toolkit Extension library
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.3)
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)

# test program
SET(SRC_POS_UNITTEST_APIS ./apis)
file(GLOB_RECURSE SRC_POS_UNITTEST_API_C ${SRC_POS_UNITTEST_APIS}/*.cpp ${SRC_POS_UNITTEST_APIS}/*.cu)

SET(SRC_POS_UNITTEST_SRC ./src)
file(GLOB_RECURSE SRC_POS_UNITTEST_SRC_C ${SRC_POS_UNITTEST_SRC}/*.cpp ${SRC_POS_UNITTEST_SRC}/*.cu)

add_executable(unittest ./main.cu ${SRC_POS_UNITTEST_API_C} ${SRC_POS_UNITTEST_SRC_C})

set(TEST_TARGETS unittest)

foreach( test_target ${TEST_TARGETS} )
    set_target_properties(${test_target} PROPERTIES LINKER_LANGUAGE CUDA)

    # target_link_libraries(${test_target} nvToolsExt -lcublas -lcurand)
    target_link_libraries(${test_target} -lcuda -lcublas -ldl -lpthread)
    # target_link_libraries(${test_target} Boost::serialization)

    target_include_directories(${test_target} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)

    # Cricket include path
    target_include_directories(${test_target} PUBLIC ../../)

    # important here to set cuda as dynamically linked
    set_target_properties(${test_target} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_features(${test_target} PUBLIC cxx_std_17)
    target_compile_options(
        ${test_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_70,code=sm_70 -gencode arch=compute_86,code=sm_86>
    )
endforeach( test_target ${TEST_TARGETS} )
