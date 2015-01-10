cmake_minimum_required(VERSION 2.8.4)
project(project_crazyflie)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(${PROJECT_NAME}_VERSION_MAJOR_0)
set(${PROJECT_NAME}_VERSION_MINOR_1)

include_directories("${PROJECT_SOURCE_DIR}/include")
find_library(USB_LIB usb-1.0)
if(${USB_LIB} STREQUAL "USB_LIB-NOTFOUND")
  message(STATUS "The USB libs appear to not be installed.")
  message(FATAL_ERROR "On Ubuntu, do: sudo apt-get install libusb-1.0-0 libusb-1.0-0-dev")
endif()

find_package( OpenCV REQUIRED )

add_library(${PROJECT_NAME}
  src/cflie/CCrazyRadio.cpp
  src/cflie/CCrazyflie.cpp
  src/cflie/CCRTPPacket.cpp
  src/cflie/CTOC.cpp)

set(SOURCE_FILES main.cpp control.cpp image_processing.cpp)
add_executable(crazyflie ${SOURCE_FILES})


target_link_libraries(${PROJECT_NAME} ${USB_LIB})
target_link_libraries(crazyflie ${PROJECT_NAME} ${OpenCV_LIBS})

install(DIRECTORY include/cflie
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  COMPONENT library
)

