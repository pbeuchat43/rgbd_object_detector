cmake_minimum_required(VERSION 3.16)
project(rgbd_object_detector_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rgbd_camera_interfaces REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Spatial.msg" "msg/Batch.msg" "msg/Status.msg"
  DEPENDENCIES rgbd_camera_interfaces vision_msgs geometry_msgs std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
