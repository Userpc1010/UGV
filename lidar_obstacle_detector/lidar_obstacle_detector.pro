QT += core
QT -= gui

TEMPLATE = lib
TARGET = lidar_obstacle_detector
DEFINES += lidar_obstacle_detector_LIBRARY

CONFIG += c++17
QMAKE_CXXFLAGS += -O3
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp
LIBS += -fopenmp

INCLUDEPATH += /usr/include/eigen3
QT_CONFIG -= no-pkg-config
CONFIG += link_pkgconfig
PKGCONFIG += eigen3

HEADERS += \
    box_fitting.hpp \
    clustering.hpp \
    config.hpp \
    ground_segmentation.hpp \
    hungarian.hpp \
    learned_detector.hpp \
    obstacle_detector.hpp \
    obstacle_detector_node.h \
    occupancy.hpp \
    patchwork/patchworkpp.h \
    patchwork/plane_fit.h \
    patchwork/types.h \
    pcl_compat.hpp \
    preprocess.hpp \
    tracker.hpp \
    types.hpp

SOURCES += \
    config.cpp \
    hungarian.cpp \
    learned_detector.cpp \
    obstacle_detector.cpp \
    obstacle_detector_node.cpp \
    occupancy.cpp \
    patchwork/patchworkpp.cpp \
    patchwork/plane_fit.cpp \
    patchworkpp_segmenter.cpp \
    tracker.cpp \
    types.cpp

unix { target.path = /usr/lib }
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += /usr/include/pcl-1.10
DEPENDPATH  += /usr/include/pcl-1.10
LIBS += -lpcl_common -lpcl_filters -lpcl_io -lpcl_kdtree \
        -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_visualization
