#include "simpleobject3d.h"

#include <QOpenGLTexture>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>

SimpleObject3D::SimpleObject3D()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_Texture(nullptr)
{
 m_Scale = 1.0f;
}

SimpleObject3D::SimpleObject3D(const QVector<VertexData> &vert, const QVector<GLuint> &ind, const QImage &img)
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_Texture(nullptr)
{
    m_Scale = 1.0f;
    init(vert, ind, img);
}

SimpleObject3D::~SimpleObject3D()
{
    free();
}

void SimpleObject3D::free()
{
    if(m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();
    if(m_IndexBuffer.isCreated()) m_IndexBuffer.destroy();
    if(m_Texture != nullptr && m_Texture->isCreated())
    { delete m_Texture; m_Texture = nullptr; }
}

void SimpleObject3D::init(const QVector<VertexData> &vert, const QVector<GLuint> &ind, const QImage &img)
{
    free();

    m_VertexBuffer.create();
    m_VertexBuffer.bind();
    m_VertexBuffer.allocate(vert.constData(), vert.size() * static_cast<int>(sizeof(VertexData)));
    m_VertexBuffer.release();// temp

    m_IndexBuffer.create();
    m_IndexBuffer.bind();
    m_IndexBuffer.allocate(ind.constData(), ind.size() * static_cast<int>(sizeof(GLuint)));
    m_IndexBuffer.release();// temp

    m_Texture = new QOpenGLTexture(img.mirrored());
    m_Texture->setMinificationFilter(QOpenGLTexture::Nearest);
    m_Texture->setMagnificationFilter(QOpenGLTexture::Linear);
    m_Texture->setWrapMode(QOpenGLTexture::Repeat);
}

void SimpleObject3D::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
    if(!m_VertexBuffer.isCreated() || !m_IndexBuffer.isCreated()) return;

    QMatrix4x4 modelMatrix;
    modelMatrix.setToIdentity();
    modelMatrix.translate(m_Translate); //
    modelMatrix.rotate(m_Rotate); // *
    modelMatrix.scale(m_Scale); // *
    modelMatrix = m_GlobalTransform * modelMatrix;

    program->setUniformValue("r_id", 0);
    program->setUniformValue("p_id", 0);
    m_Texture->bind(0);
    program->setUniformValue("u_texture", 0);
    program->setUniformValue("u_modelMatrix", modelMatrix);

    m_VertexBuffer.bind();
    m_IndexBuffer.bind();

    int offset = 0;

    int vertloc = program->attributeLocation("a_position");
    program->enableAttributeArray(vertloc);
    program->setAttributeBuffer(vertloc, GL_FLOAT, offset, 3, sizeof(VertexData));

    offset += sizeof(QVector3D);

    int texloc = program->attributeLocation("a_textcoord");
    program->enableAttributeArray(texloc);
    program->setAttributeBuffer(texloc, GL_FLOAT, offset, 2, sizeof(VertexData));

   functions->glDrawElements(GL_TRIANGLES, (m_IndexBuffer.size() / 4), GL_UNSIGNED_INT, nullptr);

    m_VertexBuffer.release();
    m_IndexBuffer.release();
    m_Texture->release();
}


void SimpleObject3D::rotate(const QQuaternion &r)
{
    m_Rotate = r;
}

void SimpleObject3D::translate(const QVector3D &t)
{
    m_Translate = t;
}

void SimpleObject3D::offset(const QVector3D &t)
{
  m_Translate += t;
}

void SimpleObject3D::scale(const float &s)
{
    m_Scale *= s;
}

void SimpleObject3D::setGlobalTransform(const QMatrix4x4 &gt)
{
    m_GlobalTransform = gt;
}
