cmake_minimum_required(VERSION 3.14)
project(rallytrack-native CXX C)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# A llama.cpp source tree (it contains ggml). Apply ggml/patches/ first to get the fast Metal conv kernel.
set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp source directory")
if (NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "pass -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE            OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL             OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON     OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

# ball tracker on ggml
add_executable(tracknet ggml/tracknet.cpp)
target_link_libraries(tracknet PRIVATE ggml)

# Metal 3x3 conv kernel vs the CPU backend
add_executable(conv2d_3x3_test ggml/tests/conv2d_3x3_test.cpp)
target_link_libraries(conv2d_3x3_test PRIVATE ggml)

# session summary with a local LLM
add_executable(coach coach/coach.cpp)
target_include_directories(coach PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_link_libraries(coach PRIVATE llama)
