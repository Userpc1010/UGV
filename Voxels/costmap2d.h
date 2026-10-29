#ifndef COSTMAP2D_H
#define COSTMAP2D_H

#include <QOpenGLBuffer>
#include <QMatrix4x4>

class QOpenGLTexture;
class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class CostmapRander
{
public:
    CostmapRander();

    ~CostmapRander();

    void init_fast();

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core * functions);

    void update_map(const QImage &img);

    void clear();

protected:

    void free();

private:

    QOpenGLBuffer m_VertexBuffer;
    QOpenGLBuffer m_IndexBuffer;
    QOpenGLTexture * m_Texture;


};

#endif // COSTMAP2D_H
