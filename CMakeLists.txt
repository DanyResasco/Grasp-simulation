cmake_minimum_required(VERSION 2.8.11)
include(ExternalProject)

set(PROJECTNAME Graspsimulation)
project(${PROJECTNAME})

# Copyright: (C) 2014 Walkman Consortium
# Authors: Enrico Mingo, Alessio Rocchi
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8.12)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0005 NEW)
    cmake_policy(SET CMP0017 NEW)
endif(COMMAND cmake_policy)
include(ExternalProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/swigmake/cmake)

find_package(Eigen3 REQUIRED COMPONENTS Dense Geometry)
find_package(CGAL REQUIRED COMPONENTS Core)

if ( EIGEN3_FOUND )
  INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
else()
    message(STATUS "This program requires the Eigen Headers, and will not be compiled.")
endif()

if ( CGAL_FOUND )
  include( ${CGAL_USE_FILE} )
  include( CGAL_CreateSingleSourceCGALProgram )
else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

INCLUDE_DIRECTORIES(src vcglib vcglib/apps/tridecimator/)

ADD_LIBRARY(test_dany_bb  SHARED src/pacman_bb.cpp
                                 src/pacman_bb_utils.cpp
                                 src/pacman_bb.hpp
                                 src/pacman_bb_utils.hpp
                                 vcglib/apps/tridecimator/mvbb_decimator.cpp
                                 vcglib/apps/tridecimator/mvbb_decimator.h
                                 vcglib/wrap/ply/plylib.cpp )

TARGET_LINK_LIBRARIES(test_dany_bb PUBLIC ${PCL_LIBRARIES} ${CGAL_LIBRARIES} ${EIGEN_LIBRARIES})

install(TARGETS test_dany_bb DESTINATION lib)

ADD_SUBDIRECTORY(python)
ADD_SUBDIRECTORY(vcglib)