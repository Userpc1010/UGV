#ifndef POINTRENDER_H
#define POINTRENDER_H

#include <QOpenGLBuffer>
#include <QMatrix4x4>

class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class PointRender
{
public:

    PointRender();
    ~PointRender();

    void init();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions, float pointsize);

    void translate(GLfloat* vertices_buffer, GLfloat* color_buffer, uint64_t counter);

protected:

    void free();

private:

    QOpenGLBuffer m_VertexBuffer;
    QOpenGLBuffer m_AttributeColorBuffer;

    uint64_t object_counter = 0;

    QVector3D color = QVector3D (1.0f, 1.0f, 1.0f);
};

#endif // POINTRENDER_H
