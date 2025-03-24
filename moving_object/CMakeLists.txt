cmake_minimum_required(VERSION 3.10)
project(moving_object)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")


find_package(OpenCV 4.1.0 REQUIRED core imgproc highgui)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

add_executable(moving_object src/main.cpp src/utils.cpp src/inc/utils.hpp)
target_link_libraries(moving_object ${OpenCV_LIBS})


find_package(GTest REQUIRED)
if(NOT GTEST_FOUND)
    message(FATAL_ERROR "Google Test not found!")
endif()

enable_testing()

add_executable(runTests tests/test_main.cpp src/utils.cpp src/inc/utils.hpp)
target_link_libraries(runTests PRIVATE GTest::gtest ${OpenCV_LIBS})

add_test(NAME moving_object_tests COMMAND runTests)