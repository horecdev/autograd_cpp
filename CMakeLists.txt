cmake_minimum_required(VERSION 3.25)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project(CalculationGraphEngine CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(tensor 
    main.cpp
    gradc/gradc.hpp
    gradc/backend/cuda/cuda_test.cu
)

# Fix: Tell the compiler to look in the root directory, so "gradc/..." paths work
target_include_directories(tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(tensor PRIVATE /fsanitize=address /Zi /W4)
    target_link_options(tensor PRIVATE /INCREMENTAL:NO)
else()
    target_compile_options(tensor PRIVATE -fsanitize=address -g -fno-omit-frame-pointer)
    target_link_options(tensor PRIVATE -fsanitize=address)
endif()