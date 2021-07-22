cmake_minimum_required (VERSION 3.11)
#cmake_minimum_required (VERSION 2.8)

string(REPLACE "-std=c++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)

project (test)

# Get the current working branch
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse --abbrev-ref HEAD
  RESULT_VARIABLE GIT_BRANCH
  OUTPUT_VARIABLE GIT_BRA OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GIT_BRANCH EQUAL 0)
    message(STATUS "On git branch version ${GIT_BRA}")
    add_definitions("-DGIT_BRA=\"${GIT_BRA}\"")
endif (GIT_BRANCH EQUAL 0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# revision version information
execute_process (
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    RESULT_VARIABLE GIT_REPO
    OUTPUT_VARIABLE GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
    )
if (GIT_REPO EQUAL 0)
    message (STATUS "Building git version ${GIT_REV}")
    add_definitions("-DGIT_REV=\"${GIT_REV}\"")
endif (GIT_REPO EQUAL 0)

set (CMAKE_CXX_FLAGS "-std=c++11 -pthread -fopenmp -O3 -lm")
#set (CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/../cmake)

find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
include_directories( ${PYTHON_INCLUDE_DIRS} )
MESSAGE(STATUS "Found PYTHON include: ${PYTHON_INCLUDE_DIRS}")
include_directories(PUBLIC "${PYTHON_NUMPY_INCLUDE_DIR}")
MESSAGE(STATUS "Found PYTHON include: ${PYTHON_NUMPY_INCLUDE_DIRS}")

find_package(CUDA)
find_package(MKL)
find_package(ESSL)
if (USE_MPI)
    find_package(MPI)
endif (USE_MPI)

if (MPI_FOUND)
    set(PYWMD pywmdmpi${PY_VER})
else (MPI_FOUND)
    set(PYWMD pywmd${PY_VER})
endif (MPI_FOUND)

include_directories( ../include )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../_simsearch/)

add_library(${PYWMD} SHARED ./src/pywmd.cpp)

if (CUDA_FOUND)
    target_link_libraries(${PYWMD} ${CUDA_LIBRARIES})
    target_link_libraries(${PYWMD} ${CUDA_CUBLAS_LIBRARIES})
    target_link_libraries(${PYWMD} ${CUDA_cusparse_LIBRARY})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA=1")
endif (CUDA_FOUND)

if (MKL_FOUND)
    target_link_libraries(${PYWMD} ${MKL_LIB1})
    target_link_libraries(${PYWMD} ${MKL_LIB2})
    target_link_libraries(${PYWMD} ${MKL_LIB3})
    target_link_libraries(${PYWMD} ${MKL_LIB4})
    target_link_libraries(${PYWMD} ${MKL_LIB5})
    target_link_libraries(${PYWMD} ${MKL_LIB6})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL=1")
endif (MKL_FOUND)

if (ESSL_FOUND)
    target_link_libraries(${PYWMD} ${ESSL_LIBRARIES})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_ESSL=1")
endif (ESSL_FOUND)

if (MPI_FOUND)
    target_link_libraries(${PYWMD} ${MPI_LIBRARIES})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI=1")
    set (CMAKE_CXX_COMPILER mpic++)
    message (STATUS "CXX compiler is ${CMAKE_CXX_COMPILER}")
endif (MPI_FOUND)

if (MPI_FOUND)
    target_link_libraries(${PYWMD} ${PROJECT_SOURCE_DIR}/../lib/libdocsimmpi.so)
else (MPI_FOUND)
    target_link_libraries(${PYWMD} ${PROJECT_SOURCE_DIR}/../lib/libdocsim.so)
endif (MPI_FOUND)

target_link_libraries(${PYWMD} ${PYTHON_LIBRARIES})

find_package (Eigen3 3.3 NO_MODULE)
if (TARGET Eigen3::Eigen)
    message (STATUS "Found Eigen.")
    target_link_libraries (${PYWMD} Eigen3::Eigen)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_EIGEN=1")
else (TARGET Eigen3::Eigen)
    message (STATUS "Could not find Eigen.")
endif (TARGET Eigen3::Eigen)

