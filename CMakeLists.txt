cmake_minimum_required(VERSION 3.15...3.26)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kspt 
                    src/pybind11.cpp 
                    src/kspt.cpp)

target_include_directories(_kspt
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(_kspt
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _kspt LIBRARY DESTINATION kspt)