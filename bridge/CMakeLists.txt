cmake_minimum_required(VERSION 3.16)
project(hw26_sim_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)  # HWauto_buff2026 用了 std::ranges
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖 ----
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# OpenVINO：默认用 /opt/intel/openvino_2024.6.0，可用 -DOPENVINO_ROOT=... 覆盖
if(NOT DEFINED OPENVINO_ROOT)
  if(EXISTS "/opt/intel/openvino_2024.6.0/runtime/cmake")
    set(OPENVINO_ROOT "/opt/intel/openvino_2024.6.0")
  endif()
endif()
if(OPENVINO_ROOT)
  set(OpenVINO_DIR "${OPENVINO_ROOT}/runtime/cmake")
endif()
find_package(OpenVINO REQUIRED)

# Zenoh C：优先 pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZENOHC REQUIRED zenohc)

# ---- HWauto_buff2026 源码根 ----
# bridge/ 就在 HWauto_buff2026 根目录下，所以源码根默认是上一级；可用 -DHW26_ROOT=... 覆盖
if(NOT DEFINED HW26_ROOT)
  get_filename_component(_default_hw26 "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
  set(HW26_ROOT "${_default_hw26}")
endif()
if(NOT EXISTS "${HW26_ROOT}/include/buff_algo/common/types.hpp")
  message(FATAL_ERROR "HW26_ROOT 看起来不对: ${HW26_ROOT}（找不到 include/buff_algo/common/types.hpp）")
endif()
message(STATUS "HW26_ROOT = ${HW26_ROOT}")

# 注意：故意【不】编译 HW26 的 src/detector/buff_detector.cpp
#       —— 它是唯一与 TensorRT/CUDA 耦合的文件，被 OpenVINO 推理替代。
set(HW26_SOURCES
  ${HW26_ROOT}/src/detector/detector_common.cpp
  ${HW26_ROOT}/src/selector/buff_selector.cpp
  ${HW26_ROOT}/src/locate/buff_pnp_solver.cpp
  ${HW26_ROOT}/src/tracker/buff_tracker.cpp
  ${HW26_ROOT}/src/tracker/tracker_status.cpp
  ${HW26_ROOT}/src/tracker/kalman_filter.cpp
  ${HW26_ROOT}/src/tracker/ransac_filter.cpp
  ${HW26_ROOT}/src/predictor/buff_predictor.cpp
)

add_executable(hw26_sim_bridge
  src/main.cpp
  src/config_io.cpp
  src/hw26_pipeline.cpp
  src/local_tracker.cpp
  ${HW26_SOURCES}
)
target_include_directories(hw26_sim_bridge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${HW26_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
  ${ZENOHC_INCLUDE_DIRS}
)
target_link_libraries(hw26_sim_bridge PRIVATE
  ${OpenCV_LIBS}
  openvino::runtime
  Eigen3::Eigen
  ${ZENOHC_LIBRARIES}
  pthread
)
target_link_directories(hw26_sim_bridge PRIVATE ${ZENOHC_LIBRARY_DIRS})
target_compile_options(hw26_sim_bridge PRIVATE -O2 -w)

# 便于直接运行：把 rpath 指到 OpenVINO 与 Zenoh 的库目录
set_target_properties(hw26_sim_bridge PROPERTIES
  BUILD_RPATH "${OPENVINO_ROOT}/runtime/lib/intel64;${ZENOHC_LIBRARY_DIRS}"
  INSTALL_RPATH "${OPENVINO_ROOT}/runtime/lib/intel64"
)

# 附带的 Zenoh 连通性探针（README 里会用到 ./build/zenoh_probe）
add_executable(zenoh_probe tools/zenoh_probe.cpp)
target_include_directories(zenoh_probe PRIVATE ${ZENOHC_INCLUDE_DIRS})
target_link_libraries(zenoh_probe PRIVATE ${ZENOHC_LIBRARIES} pthread)
target_link_directories(zenoh_probe PRIVATE ${ZENOHC_LIBRARY_DIRS})
target_compile_options(zenoh_probe PRIVATE -w)
