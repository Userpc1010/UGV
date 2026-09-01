#include "drawline.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QDebug>
#include <time.h>

//counter = 18;

//GLfloat data [] =
//{
//    -4.0, 0.0f, 0.0f,
//    4.0, 0.0f, 0.0f,

//    4.0, 0.0f, 0.0f,
//    3.0, 1.0f, 0.0f,

//    4.0, 0.0f, 0.0f,
//    3.0, -1.0f, 0.0f,

//    0.0, -4.0f, 0.0f,
//    0.0, 4.0f, 0.0f,

//    0.0, 4.0f, 0.0f,
//    1.0, 3.0f, 0.0f,

//    0.0, 4.0f, 0.0f,
//    -1.0, 3.0f, 0.0f,

//    0.0, 0.0f ,-4.0f,
//    0.0, 0.0f ,4.0f,

//    0.0, 0.0f ,4.0f,
//    0.0, 1.0f ,3.0f,

//    0.0, 0.0f ,4.0f,
//    0.0, -1.0f ,3.0f
//};

DrawLine::DrawLine()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer)
{
 free();
}

DrawLine::~DrawLine()
{
   free();
}

void DrawLine::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
    if(!m_VertexBuffer.isCreated()  ) return;

    QMatrix4x4 modelMatrix;
    modelMatrix.setToIdentity();
    program->setUniformValue("u_modelMatrix", modelMatrix);

    program->setUniformValue("r_id", 2);
    program->setUniformValue("p_id", 2);
    program->setUniformValue("u_color", color);

    m_VertexBuffer.bind();

    GLbyte atribloc = program->attributeLocation("a_position");
    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, 0, 3, 12);

    functions->glDrawArrays(GL_LINES, 0, object_counter);

    m_VertexBuffer.release();
}

void DrawLine::DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer, uint64_t counter)
{
   free();

   object_counter = counter;

   color = color_buffer;

   m_VertexBuffer.create();
   m_VertexBuffer.bind();
   m_VertexBuffer.allocate(vertices_buffer, 12 * counter);
   m_VertexBuffer.release();// temp

}

void DrawLine::free()
{
  if(m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();

  object_counter = 0;
}
