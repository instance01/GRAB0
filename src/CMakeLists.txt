cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
include(ExternalProject)
project(GRAB0)

find_program(MAKE_EXE NAMES gmake nmake make)
ExternalProject_Add(tensorboard_logger
  PREFIX            dist/
  GIT_REPOSITORY    https://github.com/Instance-contrib/tensorboard_logger.git
  GIT_TAG           convenience
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE   1
  INSTALL_COMMAND   ""
  UPDATE_COMMAND    ""
  BUILD_COMMAND     ${MAKE_EXE} all
)


message("${CMAKE_SOURCE_DIR}/dist/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/dist/libtorch")

message("${CMAKE_SOURCE_DIR}/dist/box2d/build/src")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/dist/box2d/build/src")

find_package(box2d REQUIRED)

find_package(Protobuf REQUIRED)
find_package(Torch REQUIRED)
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(dist/json)

include_directories("dist/box2d/include")

include_directories("dist/src/tensorboard_logger/include")
link_directories("dist/src/tensorboard_logger/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_definitions("-lstdc++fs")
#add_definitions("-pg")  # For gprof
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} alphazero.cpp a2c.cpp replay_buffer.cpp env_wrapper.cpp mcts.cpp tensorboard_util.cpp simple_thread_pool.cpp envs/gridworld.cpp envs/mountain_car.cpp envs/env.cpp cfg.cpp util.hpp lr_scheduler.cpp envs/cartpole.cpp gradient_bandit.cpp fuzz_bandit.cpp registry.cpp gradient_bandit_tree.cpp envs/lunar_lander.cpp gaussian_a2c.cpp gaussian_util.cpp gaussian_gradient_bandit.cpp)
target_link_libraries(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/dist/box2d/build/bin/libbox2d.a")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "stdc++fs")
target_link_libraries(${PROJECT_NAME} "-ltensorboard_logger")
target_link_libraries(${PROJECT_NAME} "-lprotobuf -pthread")
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
#target_link_libraries(${PROJECT_NAME} "-pg")  # For gprof
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
add_dependencies(${PROJECT_NAME} tensorboard_logger)
