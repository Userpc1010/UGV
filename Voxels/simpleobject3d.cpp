#include "simpleobject3d.h"

#include <QOpenGLTexture>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <oglwidgetbase.h>

SimpleObject3D::SimpleObject3D()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_Texture(nullptr)
{
 m_Scale = 1.0f;
 m_ApplyMapOffset = true;
 m_Rotate = Eigen::Quaternionf::Identity();
 m_Translate = Eigen::Vector3f::Zero();
}

SimpleObject3D::SimpleObject3D(const QVector<VertexData> &vert, const QVector<GLuint> &ind, const QImage &img)
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer),
      m_IndexBuffer(QOpenGLBuffer::IndexBuffer),
      m_Texture(nullptr)
{
    m_Scale = 1.0f;
    init(vert, ind, img);
    m_Rotate = Eigen::Quaternionf::Identity();
    m_Translate = Eigen::Vector3f::Zero();
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

    // 1. Прямой прогон через настроенный DlioFrame
    // Больше никаких локальных R_yawfix в основном коде — всё скрыто внутри!
    Eigen::Matrix3f R_odom = DlioFrame::rot(m_Rotate.toRotationMatrix());
    Eigen::Vector3f t_odom = DlioFrame::vec(m_Translate);

    // 2. Сборка локальной матрицы T
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = R_odom * m_Scale;
    T.block<3,1>(0,3) = t_odom;

    // 3. Умножение на глобальную матрицу
    Eigen::Matrix4f G = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            G(i,j) = m_GlobalTransform(i,j);

    Eigen::Matrix4f final = G * T;

    QMatrix4x4 modelMatrix(
        final(0,0), final(0,1), final(0,2), final(0,3),
        final(1,0), final(1,1), final(1,2), final(1,3),
        final(2,0), final(2,1), final(2,2), final(2,3),
        final(3,0), final(3,1), final(3,2), final(3,3)
    );

    program->setUniformValue("r_id", 0);
    program->setUniformValue("p_id", 0);
    program->setUniformValue("u_apply_map_offset", m_ApplyMapOffset ? 1 : 0);
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

void SimpleObject3D::rotate(const Eigen::Quaternionf &r)
{
  m_Rotate = r;
}

void SimpleObject3D::translate(const Eigen::Vector3f &t)
{
  m_Translate = t;
}

void SimpleObject3D::offset(const Eigen::Vector3f &t)
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
