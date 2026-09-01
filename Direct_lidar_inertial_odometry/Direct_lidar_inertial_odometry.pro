QT += core

TEMPLATE = lib
DEFINES += DLIO_LIBRARY

CONFIG += c++17

QMAKE_CXXFLAGS += -O3

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

LIBS += -fopenmp

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    LivoxData.hpp \
    dlio.h \
    livoxsdk.h \
    lsq_registration.h \
    nano_gicp.h \
    nanoflann.h \
    nanoflann_adaptor.h \
    odom.h \
    timer.hpp

SOURCES += \
    livoxsdk.cpp \
    lsq_registration.cc \
    nano_gicp.cc \
    nanoflann.cc \
    odom.cc \
    timer.cpp

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

LIBS += -L$$PWD/../../../../usr/local/lib/ -llivox_lidar_sdk_shared

LIBS += -lboost_system -lboost_thread -lboost_chrono -lboost_filesystem

LIBS += -L/usr/lib/x86_64-linux-gnu -lqhull

LIBS += -llz4

#INCLUDEPATH += $$PWD/../../../../usr/local/include/opencv4
#DEPENDPATH += $$PWD/../../../../usr/local/include/opencv4

#set package support if disabled
#QT_CONFIG -= no-pkg-config

##link opencv4 package
#CONFIG += link_pkgconfig
#PKGCONFIG += opencv4

CONFIG += link_pkgconfig
PKGCONFIG += eigen3

INCLUDEPATH += /usr/include/pcl-1.10
DEPENDPATH += /usr/include/pcl-1.10

LIBS += -lpcl_common -lpcl_filters -lpcl_io -lpcl_kdtree -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_visualization

#LIBS += -L$$PWD/../../../../usr/lib/x86_64-linux-gnu -lpcl_common -lpcl_sample_consensus -lpcl_filters -lpcl_io -lpcl_kdtree

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target


