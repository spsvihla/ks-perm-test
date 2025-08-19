cmake_minimum_required(VERSION 3.15...3.26)
project("${SKBUILD_PROJECT_NAME}" LANGUAGES CXX VERSION "${SKBUILD_PROJECT_VERSION}")

# --- Set C++ standard ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Use standard CMake build type flags ---
# Debug: no optimization, include debug symbols
# Release: full optimization
# RelWithDebInfo: optimization + debug info
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -fopenmp -funroll-loops -ffast-math")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -march=native -flto -fopenmp -funroll-loops -ffast-math")

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

# --- Python and Pybind11 ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- kspt._kspt module ---
pybind11_add_module(_kspt 
                    src/pybind11.cpp 
                    src/kspt.cpp)
target_include_directories(_kspt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(_kspt PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _kspt LIBRARY DESTINATION kspt)
