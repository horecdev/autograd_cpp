cmake_minimum_required(VERSION 3.25)
project(CalculationGraphEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tensor tensor.cpp)

if(MSVC)
    target_compile_options(tensor PRIVATE /fsanitize=address /Zi /W4)
    target_link_options(tensor PRIVATE /INCREMENTAL:NO)
else()
    target_compile_options(tensor PRIVATE -fsanitize=address -g -fno-omit-frame-pointer)
    target_link_options(tensor PRIVATE -fsanitize=address)
endif()