cmake_minimum_required(VERSION 2.8.11)
include(ExternalProject)

set(PROJECTNAME pyGraspsimulation)
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

find_package(SWIG)
if(SWIG_FOUND)
    include(UseSWIG)

    set_source_files_properties(pacman_dany_bb.i PROPERTIES CPLUSPLUS ON )
   

    set(CMAKE_${lang}_VERBOSE_FLAG "-v")
    set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
    if(NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 3.4)
     set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
    endif()
    if(NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 4.0)
     set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
    endif()
    set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
    set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
    set(CMAKE_${lang}_COMPILE_OPTIONS_SYSROOT "--sysroot=")



    include_directories(. ../src)
    link_libraries(test_dany_bb)

    set(CMAKE_SWIG_FLAGS "-Wall;-module;pydany_bb;-threads")

        set(OPENSOT_USES_PYTHON_VERSION "" CACHE STRING "Specify python version to use" )

        find_package(PythonInterp ${OPENSOT_USES_PYTHON_VERSION} REQUIRED)
        set(OPENSOT_USES_PYTHON_VERSION_ARGS)
        if(NOT OPENSOT_USES_PYTHON_VERSION)
          set (OPENSOT_USES_PYTHON_VERSION ${PYTHON_VERSION_STRING})
        endif()

        find_package(PythonLibs EXACT)
        include_directories(${PYTHON_INCLUDE_PATH})
        link_libraries(${PYTHON_LIBRARY})

        set(target_name pydany_bb)
        

        swig_add_module(${target_name} python pacman_dany_bb.i)
        set_target_properties(_${target_name} PROPERTIES OUTPUT_NAME "_pydany_bb")
        set_target_properties(_${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python)

        # installation path is determined reliably on most platforms using distutils
        execute_process(COMMAND ${PYTHON_EXECUTABLE}
            -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix=''))"
        OUTPUT_VARIABLE PYTHON_INSTDIR OUTPUT_STRIP_TRAILING_WHITESPACE )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pydany_bb.py DESTINATION ${PYTHON_INSTDIR} )
        install(TARGETS _${target_name} DESTINATION ${PYTHON_INSTDIR} )

endif(SWIG_FOUND)
