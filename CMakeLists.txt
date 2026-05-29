cmake_minimum_required(VERSION 3.25)
project(CalculationGraphEngine CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tensor 
    gradc/tensor.hpp
    main.cpp
)

# 2. Tell the compiler where to look for your headers
target_include_directories(tensor PRIVATE gradc)

if(MSVC)
    target_compile_options(tensor PRIVATE /fsanitize=address /Zi /W4)
    target_link_options(tensor PRIVATE /INCREMENTAL:NO)
else()
    target_compile_options(tensor PRIVATE -fsanitize=address -g -fno-omit-frame-pointer)
    target_link_options(tensor PRIVATE -fsanitize=address)
endif()