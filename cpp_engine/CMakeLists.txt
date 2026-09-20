cmake_minimum_required(VERSION 3.18...4.4)

# CMake >= 4.0 removed compatibility with cmake_minimum_required() versions
# below 3.5. Some FetchContent dependencies below (nlohmann_json, cpr, and
# cpr's transitive zlib fetch) still declare older minimums, which would
# otherwise hard-error during their own configure step under CMake 4.x.
# This tells CMake to treat any minimum below 3.5 declared by *those*
# projects as 3.5 instead of erroring out.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

option(ENABLE_CUDA "Enable CUDA parallel simulator (cuda_simulator target)" OFF)

if(ENABLE_CUDA)
    # --- Locate the CUDA Toolkit *before* project() enables the CUDA language ---
    #
    # On Windows the NVIDIA installer writes CUDA_PATH / CUDA_PATH_V13_4 into
    # the machine environment, but a shell or IDE that was already open before
    # install (or before the next reboot/login) still inherits an empty value.
    # MSBuild's CUDA build customization (CUDA 13.4.props) reads CudaToolkitDir
    # from that env var, so a stale shell is exactly what produces MSBuild's
    # "The CUDA Toolkit directory '' does not exist" error even though the
    # toolkit and its VS integration are installed correctly.
    #
    # Resolve a real toolkit root once, independent of the current process's
    # (possibly stale) environment, then feed it to MSBuild directly via the
    # VS toolset ("cuda=<dir>" field, below), which writes <CudaToolkitCustomDir>
    # into the generated .vcxproj -- CUDA 13.4.props prefers that over the env
    # var. On non-VS generators (e.g. Ninja) use it for CMAKE_CUDA_COMPILER instead.
    if(NOT DEFINED NBA_CUDA_TOOLKIT_ROOT)
        if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}")
            set(NBA_CUDA_TOOLKIT_ROOT "$ENV{CUDA_PATH}")
        else()
            file(GLOB _nba_cuda_candidates
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
            if(_nba_cuda_candidates)
                list(SORT _nba_cuda_candidates COMPARE NATURAL ORDER DESCENDING)
                list(GET _nba_cuda_candidates 0 NBA_CUDA_TOOLKIT_ROOT)
            endif()
            unset(_nba_cuda_candidates)
        endif()
    endif()
    set(NBA_CUDA_TOOLKIT_ROOT "${NBA_CUDA_TOOLKIT_ROOT}" CACHE PATH
        "Root of the NVIDIA CUDA Toolkit install used to configure MSBuild/nvcc")

    if(NOT NBA_CUDA_TOOLKIT_ROOT OR NOT EXISTS "${NBA_CUDA_TOOLKIT_ROOT}")
        message(FATAL_ERROR
            "ENABLE_CUDA is ON but no CUDA Toolkit install could be found. "
            "Pass -DNBA_CUDA_TOOLKIT_ROOT=<path-to-CUDA-vX.Y>, or open a fresh "
            "shell (so the CUDA_PATH env var set by the installer is visible), "
            "or re-run the CUDA installer's Visual Studio Integration step.")
    endif()

    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        # Pass the toolkit root via the VS toolset's "cuda=<dir>" field. CMake
        # writes this into the generated .vcxproj as <CudaToolkitCustomDir>,
        # which the CUDA .props/.targets use in place of $(CUDA_PATH) -- so
        # the build no longer depends on that env var being present in
        # whatever shell later invokes MSBuild.
        if(NOT CMAKE_GENERATOR_TOOLSET)
            set(CMAKE_GENERATOR_TOOLSET "cuda=${NBA_CUDA_TOOLKIT_ROOT}")
        endif()
    elseif(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${NBA_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe")
    endif()

    project(nba_spacing_engine CXX CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Explicit, stable architecture list instead of "native".
    #
    # "native" queries the GPU actually present at configure time, which is
    # fragile (CI runners / other dev machines may have no GPU, or a
    # different one, at configure time) and -- more importantly -- doesn't
    # fix the class of error this project has hit locally: "Kernel launch
    # failed: the provided PTX was compiled with an unsupported toolchain."
    #
    # That error comes from the driver's JIT compiler being asked to compile
    # embedded PTX (the forward-compatible virtual architecture) written by a
    # newer nvcc than the installed driver supports. The "-real" suffix below
    # tells nvcc to emit only a concrete SASS binary for that architecture
    # (no PTX at all), so a driver new enough to *run* the GPU never needs to
    # JIT anything -- it just loads the matching SASS directly.
    #
    # 75 = Turing (RTX 20xx / GTX 16xx), 86 = Ampere (RTX 30xx), 89 = Ada
    # (RTX 40xx, e.g. the RTX 4070 Ti this repo is developed on). Override
    # with -DCMAKE_CUDA_ARCHITECTURES=... for a different target GPU.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real")
    endif()
    message(STATUS "CUDA support enabled (toolkit: ${NBA_CUDA_TOOLKIT_ROOT}, architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    project(nba_spacing_engine CXX)
    message(STATUS "CUDA support disabled. Building CPU simulator only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static library builds to avoid dynamic library (.dll) issues on Windows
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

include(FetchContent)

# 1. Fetch nlohmann/json library
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# 2. Fetch CPR (C++ Requests) and disable system cURL usage for build stability
set(CPR_FORCE_USE_SYSTEM_CURL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.10.2
)
FetchContent_MakeAvailable(cpr)

# CPU Single-Game Narrative Simulator Executable (Always built)
add_executable(simulator main.cpp)
target_link_libraries(simulator PRIVATE nlohmann_json::nlohmann_json cpr::cpr)

# Conditional GPU Parallel Monte Carlo Simulator Executable.
#
# This is the unified pipeline: cuda_main.cpp fetches player data once, uses
# data_processor.cpp to flatten it into SoA form, runs the CPU 48-minute
# narrative simulation, then hands the same flattened data to the CUDA
# kernel runner defined in cuda_simulator.cu.
if(ENABLE_CUDA)
    add_executable(cuda_simulator
        cuda_main.cpp
        data_processor.cpp
        data_processor.hpp
        cuda_simulator.cu
        cuda_simulator.cuh
    )
    target_link_libraries(cuda_simulator PRIVATE nlohmann_json::nlohmann_json cpr::cpr)
    # -O3 only in non-Debug configs: MSVC's Debug runtime checks (/RTC1) are
    # incompatible with any optimization level, and nvcc forwards -O3 to the
    # host compiler as /O2, so applying it unconditionally breaks Debug builds.
    target_compile_options(cuda_simulator PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>)
endif()
