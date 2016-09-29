cmake_minimum_required(VERSION 2.6)
#set(CMAKE_CXX_COMPILER g++)

project(mvbb_decimator)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
include_directories( ${PROJECT_SOURCE_DIR} )


add_executable(mvbb_tridecimator  ${PROJECT_SOURCE_DIR}/apps/tridecimator/mvbb_tridecimator.cpp ${PROJECT_SOURCE_DIR}/wrap/ply/plylib.cpp)

add_library(${PROJECT_NAME}  ${PROJECT_SOURCE_DIR}/apps/tridecimator/mvbb_decimator.cpp
                             ${PROJECT_SOURCE_DIR}/apps/tridecimator/mvbb_decimator.h
                             ${PROJECT_SOURCE_DIR}/apps/tridecimator/meshdecimator.h
                             ${PROJECT_SOURCE_DIR}/wrap/ply/plylib.cpp)

