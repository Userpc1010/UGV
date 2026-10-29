#ifndef CUBERANDER_H
#define CUBERANDER_H

#include <QOpenGLBuffer>
#include <QMatrix4x4>

class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class CubeRander
{
public:
    CubeRander();

    ~CubeRander();

    void init_fast();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions, QVector3D camera);

    void translate(GLfloat* vertices_buffer, GLfloat* color_buffer, uint64_t counter);

    void clear();

protected:

    void free();

private:

    QOpenGLBuffer m_VertexBuffer;
    QOpenGLBuffer m_IndexBuffer;
    QOpenGLBuffer m_AttributePositionBuffer;
    QOpenGLBuffer m_AttributeColorBuffer;

    uint64_t object_counter = 0;
};

#endif // CUBERANDER_H
