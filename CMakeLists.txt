cmake_minimum_required(VERSION 3.5)

project(PACS LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)


include_directories(D:/opencv/build/include/)
link_directories(D:/opencv/build/x64/MinGW/lib)

if(ANDROID)
  add_library(PACS SHARED
    main.cpp
    Widget.cpp
    Widget.h
    Widget.ui
  )
else()
  add_executable(PACS
    main.cpp
    Widget.cpp
    Widget.h
    Widget.ui
  )
endif()

target_link_libraries(PACS PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(PACS PRIVATE libopencv_core3415)
target_link_libraries(PACS PRIVATE libopencv_calib3d3415)
target_link_libraries(PACS PRIVATE libopencv_dnn3415)
target_link_libraries(PACS PRIVATE libopencv_features2d3415)
target_link_libraries(PACS PRIVATE libopencv_flann3415)
target_link_libraries(PACS PRIVATE libopencv_highgui3415)
target_link_libraries(PACS PRIVATE libopencv_imgcodecs3415)
target_link_libraries(PACS PRIVATE libopencv_imgproc3415)
target_link_libraries(PACS PRIVATE libopencv_ml3415)
target_link_libraries(PACS PRIVATE libopencv_objdetect3415)
target_link_libraries(PACS PRIVATE libopencv_photo3415)
target_link_libraries(PACS PRIVATE libopencv_shape3415)
target_link_libraries(PACS PRIVATE libopencv_stitching3415)
target_link_libraries(PACS PRIVATE libopencv_superres3415)
target_link_libraries(PACS PRIVATE libopencv_video3415)
target_link_libraries(PACS PRIVATE libopencv_videoio3415)
target_link_libraries(PACS PRIVATE libopencv_videostab3415)
