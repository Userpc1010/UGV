#include "cuberander.h"

#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QDebug>
#include <time.h>


CubeRander::CubeRander()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_AttributePositionBuffer(QOpenGLBuffer::VertexBuffer),
      m_AttributeColorBuffer(QOpenGLBuffer::VertexBuffer)
{
}

CubeRander::~CubeRander()
{
   free();
}

void CubeRander::init_fast()
{
    free();

    // For cube we would need only 8 vertices but we have to
    // duplicate vertex for each face because texture coordinate
    // is different.

    GLfloat vertices_triangle_strips_f[] = {
        // Vertex data for face 0
        -0.45f, -0.45f,  0.45f,     0.0f, 0.0f, 1.0f, // v0
         0.45f, -0.45f,  0.45f,     0.0f, 0.0f, 1.0f, // v1
        -0.45f,  0.45f,  0.45f,     0.0f, 0.0f, 1.0f, // v2
         0.45f,  0.45f,  0.45f,     0.0f, 0.0f, 1.0f, // v3

        // Vertex data for face 1
        0.45f, -0.45f,  0.45f,      1.0f, 0.0f, 0.0f, // v4
        0.45f, -0.45f, -0.45f,      1.0f, 0.0f, 0.0f, // v5
        0.45f,  0.45f,  0.45f,      1.0f, 0.0f, 0.0f, // v6
        0.45f,  0.45f, -0.45f,      1.0f, 0.0f, 0.0f, // v7

        // Vertex data for face 2
         0.45f, -0.45f, -0.45f,     0.0f, 0.0f, -1.0f,  // v8
        -0.45f, -0.45f, -0.45f,     0.0f, 0.0f, -1.0f,  // v9
         0.45f,  0.45f, -0.45f,     0.0f, 0.0f, -1.0f,  // v10
        -0.45f,  0.45f, -0.45f,     0.0f, 0.0f, -1.0f,  // v11

        // Vertex data for face 3
        -0.45f, -0.45f, -0.45f,     -1.0f, 0.0f, 0.0f, // v12
        -0.45f, -0.45f,  0.45f,     -1.0f, 0.0f, 0.0f, // v13
        -0.45f,  0.45f, -0.45f,     -1.0f, 0.0f, 0.0f, // v14
        -0.45f,  0.45f,  0.45f,     -1.0f, 0.0f, 0.0f, // v15

        // Vertex data for face 4
        -0.45f, -0.45f, -0.45f,     0.0f, -1.0f, 0.0f, // v16
         0.45f, -0.45f, -0.45f,     0.0f, -1.0f, 0.0f, // v17
        -0.45f, -0.45f,  0.45f,     0.0f, -1.0f, 0.0f, // v18
         0.45f, -0.45f,  0.45f,     0.0f, -1.0f, 0.0f, // v19

        // Vertex data for face 5
        -0.45f,  0.45f,  0.45f,     0.0f, 1.0f, 0.0f, // v20
         0.45f,  0.45f,  0.45f,     0.0f, 1.0f, 0.0f, // v21
        -0.45f,  0.45f, -0.45f,     0.0f, 1.0f, 0.0f, // v22
         0.45f,  0.45f, -0.45f,     0.0f, 1.0f, 0.0f  // v23
    };

    GLubyte indices[] = {
         0,  1,  2,  3,  3,     // Face 0 - triangle strip ( v0,  v1,  v2,  v3)
         4,  4,  5,  6,  7,  7, // Face 1 - triangle strip ( v4,  v5,  v6,  v7)
         8,  8,  9, 10, 11, 11, // Face 2 - triangle strip ( v8,  v9, v10, v11)
        12, 12, 13, 14, 15, 15, // Face 3 - triangle strip (v12, v13, v14, v15)
        16, 16, 17, 18, 19, 19, // Face 4 - triangle strip (v16, v17, v18, v19)
        20, 20, 21, 22, 23      // Face 5 - triangle strip (v20, v21, v22, v23)
    };

    m_VertexBuffer.create();
    m_VertexBuffer.bind();
    m_VertexBuffer.allocate(vertices_triangle_strips_f, sizeof(vertices_triangle_strips_f));
    m_VertexBuffer.release();// temp

    m_IndexBuffer.create();
    m_IndexBuffer.bind();
    m_IndexBuffer.allocate(indices, sizeof(indices));
    m_IndexBuffer.release();// temp
}

void CubeRander::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions, QVector3D camera)
{
    if(!m_VertexBuffer.isCreated() || !m_IndexBuffer.isCreated() || !m_AttributePositionBuffer.isCreated() || !m_AttributeColorBuffer.isCreated() ) return;

    program->setUniformValue("r_id", 1);
    program->setUniformValue("p_id", 1);
    program->setUniformValue("u_lightPosition", camera);
    program->setUniformValue("u_lightPower", 1.0f);

    GLbyte atribloc = program->attributeLocation("a_position_vector");

    m_AttributePositionBuffer.bind();

    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, 0, 1, 4);
    functions->glVertexAttribDivisor(atribloc, 1);

    m_AttributePositionBuffer.release();

    atribloc = program->attributeLocation("a_color_position");

    m_AttributeColorBuffer.bind();

    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, 0, 1, 4);
    functions->glVertexAttribDivisor(atribloc, 1);

    m_AttributeColorBuffer.release();

    m_VertexBuffer.bind();
    m_IndexBuffer.bind();

    GLbyte offset = 0;

    atribloc = program->attributeLocation("a_position");
    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, offset, 3, 24);

    offset += 12;

    atribloc = program->attributeLocation("a_normal");
    program->enableAttributeArray(atribloc);
    program->setAttributeBuffer(atribloc, GL_FLOAT, offset, 3, 24);

    functions->glDrawElementsInstanced(GL_TRIANGLE_STRIP, m_IndexBuffer.size(), GL_UNSIGNED_BYTE, nullptr, object_counter);

    m_VertexBuffer.release();
    m_IndexBuffer.release();

}

void CubeRander::translate(GLfloat* vertices_buffer, GLfloat* color_buffer, uint64_t counter)
{
   if(m_AttributePositionBuffer.isCreated()) m_AttributePositionBuffer.destroy();
   if(m_AttributeColorBuffer.isCreated()) m_AttributeColorBuffer.destroy();

   object_counter = counter;

   m_AttributePositionBuffer.create();
   m_AttributePositionBuffer.bind();
   m_AttributePositionBuffer.allocate(vertices_buffer, 4 * counter);
   m_AttributePositionBuffer.release();// temp

   m_AttributeColorBuffer.create();
   m_AttributeColorBuffer.bind();
   m_AttributeColorBuffer.allocate(color_buffer, 4 * counter);
   m_AttributeColorBuffer.release();
}

void CubeRander::clear()
{
  if(m_AttributePositionBuffer.isCreated()) m_AttributePositionBuffer.destroy();
  if(m_AttributeColorBuffer.isCreated()) m_AttributeColorBuffer.destroy();

  object_counter = 0;
}


void CubeRander::free()
{
  if(m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();
  if(m_IndexBuffer.isCreated()) m_IndexBuffer.destroy();
  if(m_AttributePositionBuffer.isCreated()) m_AttributePositionBuffer.destroy();
  if(m_AttributeColorBuffer.isCreated()) m_AttributeColorBuffer.destroy();
}
