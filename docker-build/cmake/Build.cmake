get_filename_component(GS_SDF_NATIVE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GS_SDF_SUBMODULES_ROOT "${GS_SDF_NATIVE_ROOT}/submodules")

# Preserve the historical build/ layout when configuring from the repo root.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8/)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

# set(CMAKE_BUILD_TYPE "Release") set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-fPIC")

add_definitions(-O3 -DWITH_CUDA -DTHRUST_IGNORE_CUB_VERSION_CHECK)

option(ENABLE_ROS "Enable ROS support" OFF)

# Define installation directories based on build type
if(ENABLE_ROS)
  message(STATUS "ROS ENABLED - Using catkin build system")
  add_definitions(-DENABLE_ROS)

  # Use catkin build system
  find_package(
    catkin REQUIRED
    COMPONENTS roscpp
               rosbag
               roslib
               std_msgs
               geometry_msgs
               nav_msgs
               mesh_msgs
               cv_bridge
               tf)
  set(ROS_LIBRARIES ${catkin_LIBRARIES})
else()
  message(STATUS "ROS DISABLED - Using standard CMake build system")
  # No catkin dependency
  set(ROS_LIBRARIES "")
endif()

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS
    1
    CACHE INTERNAL "No dev warnings") # 关闭pcl烦人的警告
find_package(PCL REQUIRED)

# llog
add_subdirectory("${GS_SDF_SUBMODULES_ROOT}/llog"
                 "${CMAKE_CURRENT_BINARY_DIR}/submodules/llog")
include_directories("${GS_SDF_SUBMODULES_ROOT}/llog/include")

# 指定libTorch位置
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS} "${GS_SDF_NATIVE_ROOT}/include")

# Add include directories for ROS if enabled
if(ENABLE_ROS)
  include_directories(${catkin_INCLUDE_DIRS})

  catkin_package(
    # CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
    # message_runtime DEPENDS EIGEN3 PCL INCLUDE_DIRS
  )
endif()

# tcnn_binding
add_subdirectory("${GS_SDF_SUBMODULES_ROOT}/tcnn_binding"
                 "${CMAKE_CURRENT_BINARY_DIR}/submodules/tcnn_binding")
include_directories(
  "${GS_SDF_SUBMODULES_ROOT}/tcnn_binding"
  "${GS_SDF_SUBMODULES_ROOT}/tcnn_binding/submodules/tiny-cuda-nn/include"
  "${GS_SDF_SUBMODULES_ROOT}/tcnn_binding/submodules/tiny-cuda-nn/dependencies")

# kaolin_wisp_cpp
add_subdirectory("${GS_SDF_SUBMODULES_ROOT}/kaolin_wisp_cpp"
                 "${CMAKE_CURRENT_BINARY_DIR}/submodules/kaolin_wisp_cpp")
include_directories(
  "${GS_SDF_SUBMODULES_ROOT}/kaolin_wisp_cpp"
  "${GS_SDF_SUBMODULES_ROOT}/kaolin_wisp_cpp/submodules/kaolin")

add_library(
  ply_utils
  "${GS_SDF_NATIVE_ROOT}/include/utils/ply_utils/ply_utils_pcl.cpp"
  "${GS_SDF_NATIVE_ROOT}/include/utils/ply_utils/ply_utils_torch.cpp")
target_link_libraries(ply_utils ${ROS_LIBRARIES} ${PCL_LIBRARIES}
                      ${TORCH_LIBRARIES})

add_library(
  cumcubes
  "${GS_SDF_NATIVE_ROOT}/include/mesher/cumcubes/src/cumcubes.cpp"
  "${GS_SDF_NATIVE_ROOT}/include/mesher/cumcubes/src/cumcubes_kernel.cu")
target_link_libraries(cumcubes ${ROS_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(
  cumcubes
  PUBLIC "${GS_SDF_NATIVE_ROOT}/include/mesher/cumcubes/include")

add_library(mesher "${GS_SDF_NATIVE_ROOT}/include/utils/utils.cpp"
                   "${GS_SDF_NATIVE_ROOT}/include/mesher/mesher.cpp")
target_link_libraries(mesher ply_utils cumcubes)

add_library(
  data_parser
  "${GS_SDF_NATIVE_ROOT}/include/data_loader/data_parsers/base_parser.cpp")
target_link_libraries(data_parser ${OpenCV_LIBS} ${TORCH_LIBRARIES} ply_utils)

add_library(ray_utils
            "${GS_SDF_NATIVE_ROOT}/include/utils/ray_utils/ray_utils.cpp")
target_link_libraries(ray_utils ${TORCH_LIBRARIES})

add_library(data_loader "${GS_SDF_NATIVE_ROOT}/include/data_loader/data_loader.cpp"
                        "${GS_SDF_NATIVE_ROOT}/include/utils/coordinates.cpp")
target_link_libraries(data_loader data_parser ray_utils)

add_library(
  neural_net
  "${GS_SDF_NATIVE_ROOT}/include/params/params.cpp"
  "${GS_SDF_NATIVE_ROOT}/include/neural_net/sub_map.cpp"
  "${GS_SDF_NATIVE_ROOT}/include/neural_net/encoding_map.cpp"
  "${GS_SDF_NATIVE_ROOT}/include/neural_net/local_map.cpp")
target_link_libraries(neural_net mesher kaolin_wisp_cpp tcnn_binding llog
                      ray_utils)

# gsplat_cpp
add_subdirectory("${GS_SDF_SUBMODULES_ROOT}/gsplat_cpp"
                 "${CMAKE_CURRENT_BINARY_DIR}/submodules/gsplat_cpp")
include_directories("${GS_SDF_SUBMODULES_ROOT}/gsplat_cpp")

add_library(
  optimizer_utils
  "${GS_SDF_NATIVE_ROOT}/include/optimizer/optimizer_utils/optimizer_utils.cpp")
target_link_libraries(optimizer_utils ${TORCH_LIBRARIES})

include_directories("${GS_SDF_SUBMODULES_ROOT}/simple-knn")
add_library(simple-knn
            "${GS_SDF_SUBMODULES_ROOT}/simple-knn/simple_knn.cu"
            "${GS_SDF_SUBMODULES_ROOT}/simple-knn/spatial.cu")
target_link_libraries(simple-knn ${TORCH_LIBRARIES})

add_library(neural_gaussian
            "${GS_SDF_NATIVE_ROOT}/include/neural_gaussian/neural_gaussian.cpp")
target_link_libraries(neural_gaussian optimizer_utils neural_net gsplat_cpp
                      simple-knn)

add_library(loss_utils
            "${GS_SDF_NATIVE_ROOT}/include/optimizer/loss_utils/loss_utils.cpp")
target_link_libraries(loss_utils ${TORCH_LIBRARIES})

add_library(optimizer "${GS_SDF_NATIVE_ROOT}/include/optimizer/loss.cpp")
target_link_libraries(optimizer ${TORCH_LIBRARIES} loss_utils)

add_library(neural_mapping_lib
            "${GS_SDF_NATIVE_ROOT}/include/neural_mapping/neural_mapping.cpp")
target_link_libraries(neural_mapping_lib data_loader optimizer neural_net
                      neural_gaussian)

if(NOT WIN32)
  target_link_libraries(neural_mapping_lib dw)
endif()

add_executable(neural_mapping_node
               "${GS_SDF_NATIVE_ROOT}/src/neural_mapping_node.cpp")
target_link_libraries(neural_mapping_node neural_mapping_lib)
