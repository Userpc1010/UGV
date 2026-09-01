#ifndef SIMPLEOBJECT3D_H
#define SIMPLEOBJECT3D_H

#include<QOpenGLBuffer>
#include<QMatrix4x4>
#include<QVector2D>

class QOpenGLTexture;
class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

struct VertexData
{
    VertexData() {}
    VertexData(QVector3D p, QVector2D t, QVector3D n)
        : position(p), textcoord(t) {}
    QVector3D position;
    QVector2D textcoord;
};

class SimpleObject3D
{
public:
    SimpleObject3D();
    SimpleObject3D(const QVector<VertexData> &vert, const QVector<GLuint> &ind, const QImage &img);
    ~SimpleObject3D();

    void init(const QVector<VertexData> &vert, const QVector<GLuint> &ind, const QImage &img);
    void init_fast();
    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions);
    void rotate(const QQuaternion &r);
    void translate(const QVector3D &t);
    void offset (const QVector3D &t);
    void scale(const float &s);
    void setGlobalTransform(const QMatrix4x4 &gt);
    QQuaternion GetRot () {return m_Rotate;}
    QVector3D GetPosition () {return m_Translate;}



protected:
    void free();

private:
    QOpenGLBuffer m_VertexBuffer;
    QOpenGLBuffer m_IndexBuffer;
    QOpenGLTexture* m_Texture;

    QQuaternion m_Rotate;
    QVector3D m_Translate;
    float m_Scale;
    QMatrix4x4 m_GlobalTransform;
};

#endif // SIMPLEOBJECT3D_H
