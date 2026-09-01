QT += core gui
QT += opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Voxels
TEMPLATE = lib

QMAKE_CXXFLAGS += -O3

SOURCES += \
    WidgetCamera.cpp \
    WidgetCostMap.cpp \
    WidgetLidar.cpp \
    costmap2d.cpp \
        oglwidget.cpp \
    oglwidgetbase.cpp \
    pointrender.cpp \
        simpleobject3d.cpp \
    camera_3d.cpp \
    cube.cpp \
    cuberander.cpp \
    drawline.cpp \
    multidrawline.cpp

HEADERS += \
    WidgetCamera.h \
    WidgetCostMap.h \
    WidgetLidar.h \
    costmap2d.h \
        oglwidget.h \
    oglwidgetbase.h \
    pointrender.h \
        simpleobject3d.h \
    camera_3d.h \
    cube.h \
    cuberander.h \
    drawline.h \
    multidrawline.h

# Default rules for deployment.
#qnx: target.path = /tmp/$${TARGET}/bin
#else: unix:!android: target.path = /opt/$${TARGET}/bin
#!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    Recurce/shaders.qrc \
    Recurce/textures.qrc \
    Recurce/model.qrc

DISTFILES +=

