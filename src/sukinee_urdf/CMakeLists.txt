cmake_minimum_required(VERSION 3.8)
project(sukinee_urdf)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes scripts urdf rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()