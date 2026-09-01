
QT += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

QMAKE_CXXFLAGS += -O3

TARGET = UGV_Core
TEMPLATE = app

SOURCES += main.cpp \
    myserver.cpp \
    quadbike_controller.cpp \
    statelattice.cpp \
    timer.cpp \
    mainwindow.cpp \
    rgbd_camera.cpp

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    CostmapColorizer.h \
    data.h \
    myserver.h \
    quadbike_controller.h \
    statelattice.h \
    utilitiy/tools.h \
    timer.h \
    mainwindow.h \
    rgbd_camera.h

FORMS += \
    mainwindow.ui


unix:!macx: LIBS += -L$$OUT_PWD/../nav2_smac_planner/ -lnav2_smac_planner

INCLUDEPATH += $$PWD/../nav2_smac_planner
DEPENDPATH += $$PWD/../nav2_smac_planner

INCLUDEPATH += $$PWD/../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../usr/local/include

INCLUDEPATH += /usr/include/librealsense2

LIBS += -L$$DESTDIR/ -lrealsense2

unix:!macx: LIBS += -L$$PWD/../../../../usr/local/lib/ -lrealsense2-gl

unix:!macx: LIBS += -L$$OUT_PWD/../Voxels/ -lVoxels

INCLUDEPATH += $$PWD/../Voxels
DEPENDPATH += $$PWD/../Voxels

unix:!macx: LIBS += -L$$PWD/../../../usr/local/lib/ -lglfw

#unix:!macx: LIBS += -L$$OUT_PWD/../FAST-LIO2/ -lFAST-LIO2

#INCLUDEPATH += $$PWD/../FAST-LIO2
#DEPENDPATH += $$PWD/../FAST-LIO2

INCLUDEPATH += /usr/include/pcl-1.10
DEPENDPATH += /usr/include/pcl-1.10

LIBS += -lpcl_common -lpcl_filters -lpcl_io -lpcl_kdtree -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_visualization

unix:!macx: LIBS += -L$$OUT_PWD/../Direct_lidar_inertial_odometry/ -lDirect_lidar_inertial_odometry

INCLUDEPATH += $$PWD/../Direct_lidar_inertial_odometry
DEPENDPATH += $$PWD/../Direct_lidar_inertial_odometry

unix:!macx: LIBS += -L$$OUT_PWD/../Vector-Pursuit-Controller/ -lVPCController

INCLUDEPATH += $$PWD/../Vector-Pursuit-Controller
DEPENDPATH += $$PWD/../Vector-Pursuit-Controller

unix:!macx: LIBS += -L$$OUT_PWD/../MPPI-Generic/ -lMPPIGeneric

INCLUDEPATH += $$PWD/../MPPI-Generic
DEPENDPATH += $$PWD/../MPPI-Generic

INCLUDEPATH += /usr/local/include/opencv4
DEPENDPATH += /usr/local/include/opencv4

#set package support if disabled
QT_CONFIG -= no-pkg-config

#link opencv4 package
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

CONFIG += link_pkgconfig
PKGCONFIG += eigen3

# CUDA
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda-12.2
# GPU architecture (ADJUST FOR YOUR GPU)
CUDA_GENCODE  = arch=compute_86,code=sm_86
# manually add CUDA sources (ADJUST MANUALLY)
CUDA_SOURCES += cudamap.cu
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# libs used in your code
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart -lcublas -lcufft

cuda.commands        = $$CUDA_DIR/bin/nvcc -c -gencode $$CUDA_GENCODE $$NVCCFLAGS -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command  = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME} | sed \"s/^.*: //\" #For Qt 5.12.2
cuda.input           = CUDA_SOURCES
cuda.output          = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
