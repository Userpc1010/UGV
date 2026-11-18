# M-detector.pro
QT += core
QT -= gui

TEMPLATE = lib
TARGET = M_detector
DEFINES += M_detector_LIBRARY

CONFIG += c++17

QMAKE_CXXFLAGS += -O3

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp

INCLUDEPATH += /usr/include/eigen3

#set package support if disabled
QT_CONFIG -= no-pkg-config
CONFIG += link_pkgconfig
PKGCONFIG += eigen3


HEADERS += \
    MDetectorNode.h \
    \
    cluster_predict/DBSCAN_kdtree.h \
    cluster_predict/EA_disk.h \
    cluster_predict/bbox.h \
    cluster_predict/dbscan.h \
    cluster_predict/tools.hpp \
    cluster_predict/voxel_cluster.h \
    m-detector/DynObjCluster.h \
    m-detector/DynObjFilter.h \
   parallel_q.h \
   types.h

SOURCES += \
    DynObjFilter.cpp \
    MDetectorNode.cpp \
     \
    DynObjCluster.cpp
    
# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += /usr/include/pcl-1.10
DEPENDPATH += /usr/include/pcl-1.10

LIBS += -lpcl_common -lpcl_filters -lpcl_io -lpcl_kdtree -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_visualization


