cmake_minimum_required(VERSION 3.18)  # 3.18+ for better CUDA/HIP support
project(spmvBenchmark LANGUAGES CXX)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(COMMON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

# Find OpenMP (required for CPU parallelism)
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------
# Detect available GPU backends (do NOT enable them yet)
# ------------------------------------------------------------------

# --- CUDA detection ---
find_program(CMAKE_CUDA_COMPILER nvcc DOC "CUDA compiler")
if(CMAKE_CUDA_COMPILER)
    set(HAS_CUDA TRUE)
    message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER}")
else()
    set(HAS_CUDA FALSE)
    message(WARNING "CUDA compiler (nvcc) not found – CUDA targets disabled")
endif()

# --- HIP detection ---
find_program(HIPCC_COMPILER hipcc DOC "HIP compiler")
if(HIPCC_COMPILER)
    set(HAS_HIP TRUE)
    message(STATUS "Found HIP compiler: ${HIPCC_COMPILER}")
else()
    set(HAS_HIP FALSE)
    message(WARNING "HIP compiler (hipcc) not found – HIP targets disabled")
endif()

# ------------------------------------------------------------------
# Collect and classify source files
# ------------------------------------------------------------------

# Explicitly restrict to project source dir
file(GLOB_RECURSE ALL_SOURCES
    CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cxx
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/*.hip
)

set(CPU_SOURCES "")
set(CUDA_SOURCES "")
set(HIP_SOURCES "")

foreach(source_file IN LISTS ALL_SOURCES)
    get_filename_component(ext "${source_file}" EXT)
    if(ext STREQUAL ".cu")
        list(APPEND CUDA_SOURCES "${source_file}")
    elseif(ext STREQUAL ".hip")
        list(APPEND HIP_SOURCES "${source_file}")
    else()
        list(APPEND CPU_SOURCES "${source_file}")
    endif()
endforeach()

# ------------------------------------------------------------------
# Build options (user-controllable via -D)
# ------------------------------------------------------------------

option(BUILD_CPU_ONLY     "Build CPU-only executable"      ON)
option(BUILD_GPU_SUPPORT  "Build GPU-specific executables" ON)
option(BUILD_UNIFIED      "Build unified CPU+GPU executable" ON)

# ------------------------------------------------------------------
# Target list for installation
# ------------------------------------------------------------------

set(INSTALLABLE_TARGETS "")

# ------------------------------------------------------------------
# 1. CPU-ONLY TARGET (pure C++, no GPU involvement)
# ------------------------------------------------------------------

if(BUILD_CPU_ONLY)
    add_executable(${PROJECT_NAME}_cpu ${CPU_SOURCES})
    target_include_directories(${PROJECT_NAME}_cpu PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_cpu PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME}_cpu PRIVATE
        -Wall -Wextra -O3 -DNDEBUG
    )
    target_compile_definitions(${PROJECT_NAME}_cpu PRIVATE CPU_ONLY=1)
    add_custom_target(cpu_build DEPENDS ${PROJECT_NAME}_cpu)
    list(APPEND INSTALLABLE_TARGETS ${PROJECT_NAME}_cpu)
endif()

# ------------------------------------------------------------------
# 2. CUDA-SPECIFIC TARGET (only if CUDA available and requested)
# ------------------------------------------------------------------

if(BUILD_GPU_SUPPORT AND HAS_CUDA AND CUDA_SOURCES)
    # Enable CUDA language only when needed
    enable_language(CUDA)

    add_executable(${PROJECT_NAME}_cuda ${CPU_SOURCES} ${CUDA_SOURCES})
    target_include_directories(${PROJECT_NAME}_cuda PRIVATE ${COMMON_INCLUDE_DIRS})
    set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
        CUDA_ARCHITECTURES "75;80;86"  # Adjust as needed
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${PROJECT_NAME}_cuda PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${PROJECT_NAME}_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 --compiler-options -Wall>
    )
    target_compile_definitions(${PROJECT_NAME}_cuda PRIVATE
        CUDA_ENABLED=1
        CPU_IMPLEMENTATION=1
        GPU_IMPLEMENTATION=1
    )
    add_custom_target(cuda_build DEPENDS ${PROJECT_NAME}_cuda)
    list(APPEND INSTALLABLE_TARGETS ${PROJECT_NAME}_cuda)
endif()

# ------------------------------------------------------------------
# 3. HIP-SPECIFIC TARGET (only if HIP available and requested)
# ------------------------------------------------------------------

if(BUILD_GPU_SUPPORT AND HAS_HIP AND HIP_SOURCES)
    # Only now find HIP package (avoid polluting CPU build)
    find_package(hip REQUIRED)

    add_executable(${PROJECT_NAME}_hip ${CPU_SOURCES} ${HIP_SOURCES})
    target_include_directories(${PROJECT_NAME}_hip PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_hip PRIVATE OpenMP::OpenMP_CXX hip::host)
    target_compile_options(${PROJECT_NAME}_hip PRIVATE
        -Wall -Wextra -O3 -DNDEBUG
    )
    target_compile_definitions(${PROJECT_NAME}_hip PRIVATE
        HIP_ENABLED=1
        CPU_IMPLEMENTATION=1
        GPU_IMPLEMENTATION=1
    )
    add_custom_target(hip_build DEPENDS ${PROJECT_NAME}_hip)
    list(APPEND INSTALLABLE_TARGETS ${PROJECT_NAME}_hip)
endif()

# ------------------------------------------------------------------
# 4. UNIFIED TARGET (CPU + all available GPU backends)
# ------------------------------------------------------------------

if(BUILD_UNIFIED AND (HAS_CUDA OR HAS_HIP))
    set(UNIFIED_SOURCES ${CPU_SOURCES})

    if(HAS_CUDA AND CUDA_SOURCES)
        list(APPEND UNIFIED_SOURCES ${CUDA_SOURCES})
        enable_language(CUDA)  # Ensure CUDA is enabled
    endif()

    if(HAS_HIP AND HIP_SOURCES)
        list(APPEND UNIFIED_SOURCES ${HIP_SOURCES})
        find_package(hip REQUIRED)
    endif()

    add_executable(${PROJECT_NAME}_unified ${UNIFIED_SOURCES})
    target_include_directories(${PROJECT_NAME}_unified PRIVATE ${COMMON_INCLUDE_DIRS})

    # Link libraries
    target_link_libraries(${PROJECT_NAME}_unified PRIVATE OpenMP::OpenMP_CXX)
    if(HAS_HIP)
        target_link_libraries(${PROJECT_NAME}_unified PRIVATE hip::host)
    endif()

    # Compile options
    target_compile_options(${PROJECT_NAME}_unified PRIVATE
        -Wall -Wextra -O3 -DNDEBUG
    )
    if(HAS_CUDA)
        target_compile_options(${PROJECT_NAME}_unified PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
        )
        set_target_properties(${PROJECT_NAME}_unified PROPERTIES
            CUDA_ARCHITECTURES "75;80;86"
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()

    # Definitions
    target_compile_definitions(${PROJECT_NAME}_unified PRIVATE
        CPU_IMPLEMENTATION=1
        GPU_IMPLEMENTATION=1
        UNIFIED_BUILD=1
        $<$<BOOL:${HAS_CUDA}>:CUDA_ENABLED=1>
        $<$<BOOL:${HAS_HIP}>:HIP_ENABLED=1>
    )

    add_custom_target(unified_build DEPENDS ${PROJECT_NAME}_unified)
    list(APPEND INSTALLABLE_TARGETS ${PROJECT_NAME}_unified)
endif()

# ------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------

if(TARGET ${PROJECT_NAME}_cpu)
    add_custom_target(${PROJECT_NAME} DEPENDS ${PROJECT_NAME}_cpu)
elseif(TARGET ${PROJECT_NAME}_unified)
    add_custom_target(${PROJECT_NAME} DEPENDS ${PROJECT_NAME}_unified)
else()
    message(WARNING "No default target could be created.")
endif()

# ------------------------------------------------------------------
# Installation (only existing targets)
# ------------------------------------------------------------------

if(INSTALLABLE_TARGETS)
    install(TARGETS ${INSTALLABLE_TARGETS} RUNTIME DESTINATION bin)
endif()

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

message("")
message("==================================================")
message("Build Configuration Summary")
message("==================================================")
message("Project:              ${PROJECT_NAME}")
message("Build Type:           ${CMAKE_BUILD_TYPE}")
message("C++ Standard:         ${CMAKE_CXX_STANDARD}")
message("OpenMP:               ${OpenMP_CXX_FOUND}")
message("CUDA Support:         ${HAS_CUDA}")
# message(STATUS "CUDA_SOURCES = ${CUDA_SOURCES}") // check if have .cu files
message("HIP Support:          ${HAS_HIP}")
# message(STATUS "HIP_SOURCES = ${HIP_SOURCES}")   // check if have .hip files
message("")
message("Requested Builds:")
message("  CPU Only:           ${BUILD_CPU_ONLY}")
message("  GPU Specific:       ${BUILD_GPU_SUPPORT}")
message("  Unified:            ${BUILD_UNIFIED}")
message("")
message("Available Targets:")
if(BUILD_CPU_ONLY)
    message("  - ${PROJECT_NAME}_cpu   (make cpu_build)")
endif()
if(BUILD_GPU_SUPPORT AND HAS_CUDA AND CUDA_SOURCES)
    message("  - ${PROJECT_NAME}_cuda  (make cuda_build)")
endif()
if(BUILD_GPU_SUPPORT AND HAS_HIP AND HIP_SOURCES)
    message("  - ${PROJECT_NAME}_hip   (make hip_build)")
endif()
if(BUILD_UNIFIED AND (HAS_CUDA OR HAS_HIP))
    message("  - ${PROJECT_NAME}_unified (make unified_build)")
endif()
message("Default 'make' target: ${PROJECT_NAME}")
message("==================================================")
message("")