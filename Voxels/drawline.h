#ifndef DRAWLINE_H
#define DRAWLINE_H

#include <QOpenGLBuffer>
#include <QMatrix4x4>

class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class DrawLine
{
public:
    DrawLine();

    ~DrawLine();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions);

    void DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer, uint64_t counter);

protected:

    void free();

private:

    QOpenGLBuffer m_VertexBuffer;

    uint32_t object_counter = 0;

    QVector3D color;

};

#endif // DRAWLINE_H
