#ifndef DRAWLINE_H
#define DRAWLINE_H

#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include <Eigen/Core>
#include <Eigen/Geometry>

class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class DrawLine
{
public:
    DrawLine();

    ~DrawLine();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions);

    void DrawLines(GLfloat* vertices_buffer, Eigen::Vector3f color_buffer, uint64_t counter);

protected:

    void free();

private:

    QOpenGLBuffer m_VertexBuffer;

    uint32_t object_counter = 0;

    Eigen::Vector3f color = Eigen::Vector3f::Ones();

};

#endif // DRAWLINE_H
