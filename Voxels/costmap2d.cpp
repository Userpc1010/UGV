#include "costmap2d.h"

#include <QOpenGLTexture>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <QDebug>
#include <time.h>


CostmapRander::CostmapRander()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_Texture(nullptr)
{
}

CostmapRander::~CostmapRander()
{
   free();
}

void CostmapRander::init_fast()
{
    free();

    GLfloat vertices_triangle_strips_f[] = {
        // position.x, position.y, position.z, texcoord.x, texcoord.y
         200.0f, 0.0f, -200.0f,  1.0f, 0.0f,  // Вершина 0: была 1
        -200.0f, 0.0f, -200.0f,  0.0f, 0.0f,  // Вершина 1: была 0
         200.0f, 0.0f,  200.0f,  1.0f, 1.0f,  // Вершина 2: была 3
        -200.0f, 0.0f,  200.0f,  0.0f, 1.0f   // Вершина 3: была 2
    };

    // Индексы
    GLubyte indices[] = {0, 1, 2, 3};

    m_VertexBuffer.create();
    m_VertexBuffer.bind();
    m_VertexBuffer.allocate(vertices_triangle_strips_f, sizeof(vertices_triangle_strips_f));
    m_VertexBuffer.release();// temp

    m_IndexBuffer.create();
    m_IndexBuffer.bind();
    m_IndexBuffer.allocate(indices, sizeof(indices));
    m_IndexBuffer.release();// temp

}

void CostmapRander::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
    if(!m_VertexBuffer.isCreated() || !m_IndexBuffer.isCreated() || !m_Texture || !m_Texture->isCreated()) return;

    program->setUniformValue("r_id", 0);
    program->setUniformValue("p_id", 0);
    m_Texture->bind(0);
    program->setUniformValue("u_texture", 0);

    m_VertexBuffer.bind();
    m_IndexBuffer.bind();

    GLbyte offset = 0;

    GLbyte vertloc = program->attributeLocation("a_position");
    program->enableAttributeArray(vertloc);
    program->setAttributeBuffer(vertloc, GL_FLOAT, offset, 3, 20);

    offset += 12;

    GLbyte texloc = program->attributeLocation("a_textcoord");
    program->enableAttributeArray(texloc);
    program->setAttributeBuffer(texloc, GL_FLOAT, offset, 2, 20);

    functions->glDrawElements(GL_TRIANGLE_STRIP, m_IndexBuffer.size(), GL_UNSIGNED_BYTE, nullptr);

    m_VertexBuffer.release();
    m_IndexBuffer.release();
    m_Texture->release();
}

void CostmapRander::update_map(const QImage &img)
{
  if(m_Texture != nullptr && m_Texture->isCreated()) { delete m_Texture; m_Texture = nullptr; }

 m_Texture = new QOpenGLTexture(img.mirrored());
 m_Texture->setMinificationFilter(QOpenGLTexture::Nearest);
 m_Texture->setMagnificationFilter(QOpenGLTexture::Nearest);
 m_Texture->setWrapMode(QOpenGLTexture::ClampToEdge);

}

void CostmapRander::clear()
{
  if(m_Texture != nullptr && m_Texture->isCreated())
  { delete m_Texture; m_Texture = nullptr; }
}

void CostmapRander::free()
{
  if(m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();
  if(m_IndexBuffer.isCreated()) m_IndexBuffer.destroy();
  if(m_Texture != nullptr && m_Texture->isCreated())
  { delete m_Texture; m_Texture = nullptr; }
}
