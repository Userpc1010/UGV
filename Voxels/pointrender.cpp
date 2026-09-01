#include "pointrender.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QDebug>

PointRender::PointRender()
  : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
   m_AttributeColorBuffer(QOpenGLBuffer::VertexBuffer)
{

}

PointRender::~PointRender()
{
    free();
}

void PointRender::init()
{
   free(); //?
}

void PointRender::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions, float pointsize)
{
    if(!m_AttributeColorBuffer.isCreated() || !m_VertexBuffer.isCreated())  return;

    QMatrix4x4 modelMatrix;
    modelMatrix.setToIdentity();
    program->setUniformValue("u_modelMatrix", modelMatrix);

    program->setUniformValue("r_id", 3);
    program->setUniformValue("p_id", 3);

    GLbyte atribloc = program->attributeLocation("color_position");

    m_AttributeColorBuffer.bind();

    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, 0, 3, 12);

    m_AttributeColorBuffer.release();

    m_VertexBuffer.bind();

    atribloc = program->attributeLocation("a_position");

    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, 0, 3, 12);

    functions->glPointSize(pointsize);

    functions->glDrawArrays(GL_POINTS, 0, object_counter);

    m_VertexBuffer.release();

}

void PointRender::translate(GLfloat* vertices_buffer, GLfloat* color_buffer, uint64_t counter)
{
    free();

    object_counter = counter;

    m_VertexBuffer.create();
    m_VertexBuffer.bind();
    m_VertexBuffer.allocate(vertices_buffer, 12 * counter);
    m_VertexBuffer.release();// temp

    m_AttributeColorBuffer.create();
    m_AttributeColorBuffer.bind();
    m_AttributeColorBuffer.allocate(color_buffer, 12 * counter);
    m_AttributeColorBuffer.release();
}

void PointRender::free()
{
    if(m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();
    if(m_AttributeColorBuffer.isCreated()) m_AttributeColorBuffer.destroy();

    object_counter = 0;
}
