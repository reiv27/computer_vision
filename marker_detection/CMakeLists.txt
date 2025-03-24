cmake_minimum_required(VERSION 3.10)
project(marker_detection)

find_package(OpenCV 4.1.0 REQUIRED core imgproc highgui)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

find_package(GTest REQUIRED)
if(NOT GTEST_FOUND)
    message(FATAL_ERROR "Google Test not found!")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -march=native -Ofast")

include_directories(src/inc)

add_executable(marker_detection src/main.cpp src/utils.cpp)
target_link_libraries(marker_detection ${OpenCV_LIBS})

enable_testing()

add_executable(runTests tests/test_main.cpp src/utils.cpp)
target_link_libraries(runTests PRIVATE GTest::gtest ${OpenCV_LIBS})

add_test(NAME marker_detection_tests COMMAND runTests)