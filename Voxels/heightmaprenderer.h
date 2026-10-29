#pragma once
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <cstdint>

class QOpenGLShaderProgram;
class QOpenGLFunctions_4_5_Core;

class HeightMapRenderer
{
public:
    HeightMapRenderer();
    ~HeightMapRenderer();

    void init_fast();
    void update_map(const float* data, float min, float max);   // 400*200 float, упаковано
    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core* functions, QVector3D camera);

protected:
    void free();

private:

    float h_max;
    float h_min;

    QOpenGLBuffer   m_VertexBuffer;
    QOpenGLTexture* m_HeightTexture = nullptr;
    int m_VertexCount = 0;
};
