#include "heightmaprenderer.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_5_Core>
#include <vector>

static constexpr int kGridSize = 400;   // 400x400 высот
static constexpr int kTexW     = 200;   // 400 / 2
static constexpr int kTexH     = 400;

HeightMapRenderer::HeightMapRenderer()
    : m_VertexBuffer(QOpenGLBuffer::VertexBuffer)
{}

HeightMapRenderer::~HeightMapRenderer() { free(); }

void HeightMapRenderer::init_fast()
{
    free();

    std::vector<float> verts;
    verts.reserve((kGridSize - 1) * (kGridSize - 1) * 6 * 6);
    const float half = 200.0f;
    const float step = 400.0f / (kGridSize - 1);

    // Коэффициент зазора (0.05 = 5% от размера ячейки)
    // Чем больше, тем толще черная сетка
    const float gap = 0.05f * step;

    float initheight = 0.0f;

    for (int row = 0; row < kGridSize - 1; ++row) {
        for (int col = 0; col < kGridSize - 1; ++col) {

            // Углы квада (сжатые к центру на величину gap)
            float x0 = -half + col * step + gap;
            float x1 = -half + (col + 1) * step - gap;
            float z0 =  half - row * step - gap;
            float z1 =  half - (row + 1) * step + gap;

            // UV координаты оставляем как были (нам они нужны для текстуры высот)
            float u0 = (float(col / 2) + 0.5f) / float(kTexW);
            float u1 = (float((col + 1) / 2) + 0.5f) / float(kTexW);
            float v0 = (float(row) + 0.5f) / float(kTexH);
            float v1 = (float(row + 1) + 0.5f) / float(kTexH);
            float p0 = float(col % 2);
            float p1 = float((col + 1) % 2);

            // 6 вершин (2 треугольника) для сжатого квада
            // Ось Y = 0, высота добавится в шейдере
            float quad[6][6] = {
                { x0, initheight, z0, u0, v0, p0 },
                { x1, initheight, z0, u1, v0, p1 },
                { x0, initheight, z1, u0, v1, p0 },
                { x1, initheight, z0, u1, v0, p1 },
                { x1, initheight, z1, u1, v1, p1 },
                { x0, initheight, z1, u0, v1, p0 },
            };

            for (auto& v : quad) {
                for (int k = 0; k < 6; ++k) verts.push_back(v[k]);
            }
        }
    }
    m_VertexCount = int(verts.size() / 6);

    m_VertexBuffer.create();
    m_VertexBuffer.bind();
    m_VertexBuffer.allocate(verts.data(), int(verts.size() * sizeof(float)));
    m_VertexBuffer.release();

    m_HeightTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_HeightTexture->setSize(kTexW, kTexH);
    m_HeightTexture->setFormat(QOpenGLTexture::R32F);
    m_HeightTexture->setMipLevels(1);
    m_HeightTexture->allocateStorage();
    m_HeightTexture->setMinificationFilter(QOpenGLTexture::Nearest);
    m_HeightTexture->setMagnificationFilter(QOpenGLTexture::Nearest);
    m_HeightTexture->setWrapMode(QOpenGLTexture::ClampToEdge);
}

void HeightMapRenderer::update_map(const float* data, float min, float max)
{
    if (!m_HeightTexture || !m_HeightTexture->isCreated() || !data) return;

    h_max = max; h_min = min;

    m_HeightTexture->bind();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kTexW, kTexH,
                    GL_RED, GL_FLOAT, data);
    m_HeightTexture->release();
}

void HeightMapRenderer::draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core* functions, QVector3D camera)
{
    if (!m_VertexBuffer.isCreated() || !m_HeightTexture || !m_HeightTexture->isCreated() || m_VertexCount == 0) return;

    QMatrix4x4 modelMatrix;
    modelMatrix.setToIdentity();
    program->setUniformValue("u_modelMatrix", modelMatrix);

    program->setUniformValue("p_id", 4);
    program->setUniformValue("r_id", 4);
    program->setUniformValue("u_apply_map_offset", 0);
    program->setUniformValue("u_hMin", h_min);
    program->setUniformValue("u_hMax", h_max);
    program->setUniformValue("u_lightPosition", camera.normalized());
    program->setUniformValue("u_lightPower", 1.5f); // Силу света можно подкрутить

    m_HeightTexture->bind(0);
    program->setUniformValue("u_heightTexture", 0);

    m_VertexBuffer.bind();
    int stride = 6 * sizeof(float);
    int loc = program->attributeLocation("a_position");
    program->enableAttributeArray(loc);
    program->setAttributeBuffer(loc, GL_FLOAT, 0, 3, stride);
    loc = program->attributeLocation("a_textcoord");
    program->enableAttributeArray(loc);
    program->setAttributeBuffer(loc, GL_FLOAT, 3 * sizeof(float), 2, stride);
    loc = program->attributeLocation("a_parity");
    program->enableAttributeArray(loc);
    program->setAttributeBuffer(loc, GL_FLOAT, 5 * sizeof(float), 1, stride);

    functions->glDrawArrays(GL_TRIANGLES, 0, m_VertexCount);

    m_VertexBuffer.release();
    m_HeightTexture->release();
}

void HeightMapRenderer::free()
{
    if (m_VertexBuffer.isCreated()) m_VertexBuffer.destroy();
    if (m_HeightTexture) { delete m_HeightTexture; m_HeightTexture = nullptr; }
    m_VertexCount = 0;
}
