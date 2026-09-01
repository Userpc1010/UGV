QT += core

TEMPLATE = lib
DEFINES += FASTLIO2_LIBRARY

CONFIG += c++17

QMAKE_CXXFLAGS += -O3
QMAKE_CXXFLAGS += -fexceptions

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.


# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    IKFoM_toolkit/esekfom/esekfom.cpp \
    IMU_Processing.cpp \
    ikd-Tree/ikd_Tree.cpp \
    lidarmapping.cpp \
    preprocess.cpp \
    timer.cpp

HEADERS += \
    Exp_mat.hpp \
    IKFoM_toolkit/esekfom/esekfom.hpp \
    IKFoM_toolkit/esekfom/util.hpp \
    IKFoM_toolkit/mtk/build_manifold.hpp \
    IKFoM_toolkit/mtk/src/SubManifold.hpp \
    IKFoM_toolkit/mtk/src/mtkmath.hpp \
    IKFoM_toolkit/mtk/src/vectview.hpp \
    IKFoM_toolkit/mtk/startIdx.hpp \
    IKFoM_toolkit/mtk/types/S2.hpp \
    IKFoM_toolkit/mtk/types/SOn.hpp \
    IKFoM_toolkit/mtk/types/vect.hpp \
    IKFoM_toolkit/mtk/types/wrapped_cv_mat.hpp \
    IMU_Processing.hpp \
    LivoxData.hpp \
    common_lib.hpp \
    ikd-Tree/ikd_Tree.hpp \
    lidarmapping.hpp \
    preprocess.hpp \
    so3_math.hpp \
    timer.hpp

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -llivox_lidar_sdk_shared

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

#INCLUDEPATH += $$PWD/../../../../usr/local/include/opencv4
#DEPENDPATH += $$PWD/../../../../usr/local/include/opencv4

#INCLUDEPATH += $$PWD/../../../../usr/local/include/eigen3

LIBS += -lboost_system -lboost_thread -lboost_chrono -lboost_filesystem

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

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target
