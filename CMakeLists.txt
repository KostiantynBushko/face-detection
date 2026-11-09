cmake_minimum_required(VERSION 3.20)

project(face-detection
    VERSION 0.1.0
    DESCRIPTION "Face detection and recognition for camera streams (YuNet + SFace)"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Build-time options.
#
# When this project is consumed as a submodule of objective-control it is
# added with add_subdirectory(); PROJECT_IS_TOP_LEVEL is then FALSE and the
# apps/tests are not built unless the parent asks for them.
# ---------------------------------------------------------------------------
option(FACE_DETECTION_ENABLE_OPENCV "Enable the OpenCV-backed face stack" ON)
option(FACE_DETECTION_BUILD_APPS    "Build the example application"       ${PROJECT_IS_TOP_LEVEL})
option(FACE_DETECTION_BUILD_TESTS   "Build unit tests"                     ${PROJECT_IS_TOP_LEVEL})
option(FACE_DETECTION_ENABLE_LEGACY "Enable the contrib 'face' module (LBPH fallback)" OFF)
option(FACE_DETECTION_ENABLE_SANITIZERS "Build with Address/Undefined sanitizers (debug)" OFF)

# ---------------------------------------------------------------------------
# Global C++ settings.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Use CMAKE_CURRENT_LIST_DIR so the paths stay correct when this
# project is consumed via add_subdirectory() from another project.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Dependencies.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Warnings.cmake)

# ---------------------------------------------------------------------------
# Subdirectories.
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(FACE_DETECTION_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(FACE_DETECTION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
