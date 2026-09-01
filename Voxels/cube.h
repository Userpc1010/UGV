#ifndef CUBE_H
#define CUBE_H
#include "transformational.h"
#include "simpleobject3d.h"
#include <QQuaternion>
#include <QVector3D>
#include <QMatrix4x4>

class Cube
{
public:
    Cube();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core *functions);
    void draw_cube(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core *functions);
    void initCube(float width, const QImage &img);

    void add_cube (QVector3D pos);
    void translation(QVector3D pos);
    void Invisible_AllCube (bool invisible_);
    void Offset_AllCube (QVector3D pos);
    void delete_cube ();

private:

bool invisible;

QImage* cube;

QVector<SimpleObject3D*> m_Objects_Cube;

};

#endif // CUBE_H
