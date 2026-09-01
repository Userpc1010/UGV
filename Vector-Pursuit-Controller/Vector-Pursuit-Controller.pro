# Vector-Pursuit-Controller.pro
QT -= gui

TEMPLATE = lib
TARGET = VPCController
DEFINES += VPCCONTROLLER_LIBRARY

CONFIG += c++17

QMAKE_CXXFLAGS += -O3

INCLUDEPATH += /usr/include/eigen3
INCLUDEPATH += $$PWD/..

#set package support if disabled
QT_CONFIG -= no-pkg-config
CONFIG += link_pkgconfig
PKGCONFIG += eigen3

HEADERS += \
    vector_pursuit_controller.hpp \
    vector_pursuit_controller_collision_checker.hpp

SOURCES += \
    vector_pursuit_controller.cpp \
    vector_pursuit_controller_collision_checker.cpp
    
# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$$OUT_PWD/../nav2_smac_planner/ -lnav2_smac_planner

INCLUDEPATH += $$PWD/../nav2_smac_planner
DEPENDPATH += $$PWD/../nav2_smac_planner
