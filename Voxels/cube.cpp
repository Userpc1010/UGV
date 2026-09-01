#include "cube.h"
#include <QImage>

Cube::Cube()
{
  cube = new QImage(":/Target.png");

  invisible = false;
}

void Cube::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
   if(!invisible) for(auto o: m_Objects_Cube) o->draw(program, functions);
}

void Cube::initCube(float width, const QImage &img)
{
    float width_div_2 = width / 2.0f;

    QVector<VertexData> vertexes;
    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, width_div_2), QVector2D(0.0f, 1.0f), QVector3D(0.0f, 0.0f, 1.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, width_div_2), QVector2D(0.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, width_div_2), QVector2D(1.0f, 1.0f), QVector3D(0.0f, 0.0f, 1.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, width_div_2), QVector2D(1.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f)));

    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, width_div_2),QVector2D(0.0f, 1.0f), QVector3D(1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, width_div_2), QVector2D(0.0f, 0.0f), QVector3D(1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, -width_div_2), QVector2D(1.0f, 1.0f), QVector3D(1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, -width_div_2), QVector2D(1.0f, 0.0f), QVector3D(1.0f, 0.0f, 0.0f)));

    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, width_div_2), QVector2D(0.0f, 1.0f), QVector3D(0.0f, 1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, -width_div_2), QVector2D(0.0f, 0.0f), QVector3D(0.0f, 1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, width_div_2), QVector2D(1.0f, 1.0f), QVector3D(0.0f, 1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, -width_div_2), QVector2D(1.0f, 0.0f), QVector3D(0.0f, 1.0f, 0.0f)));

    vertexes.append(VertexData(QVector3D(width_div_2, width_div_2, -width_div_2), QVector2D(0.0f, 1.0f), QVector3D(0.0f, 0.0f, -1.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, -width_div_2), QVector2D(0.0f, 0.0f), QVector3D(0.0f, 0.0f, -1.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, -width_div_2), QVector2D(1.0f, 1.0f), QVector3D(0.0f, 0.0f, -1.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, -width_div_2), QVector2D(1.0f, 0.0f), QVector3D(0.0f, 0.0f, -1.0f)));

    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, width_div_2), QVector2D(0.0f, 1.0f), QVector3D(-1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, width_div_2, -width_div_2), QVector2D(0.0f, 0.0f), QVector3D(-1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, width_div_2), QVector2D(1.0f, 1.0f), QVector3D(-1.0f, 0.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, -width_div_2), QVector2D(1.0f, 0.0f), QVector3D(-1.0f, 0.0f, 0.0f)));

    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, width_div_2), QVector2D(0.0f, 1.0f), QVector3D(0.0f, -1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(-width_div_2, -width_div_2, -width_div_2), QVector2D(0.0f, 0.0f), QVector3D(0.0f, -1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, width_div_2), QVector2D(1.0f, 1.0f), QVector3D(0.0f, -1.0f, 0.0f)));
    vertexes.append(VertexData(QVector3D(width_div_2, -width_div_2, -width_div_2), QVector2D(1.0f, 0.0f), QVector3D(0.0f, -1.0f, 0.0f)));

    QVector<GLuint> indexes;
    for(GLuint i = 0; i < 24; i += 4)
    {
        indexes.append(i + 0); indexes.append(i + 1); indexes.append(i + 2);
        indexes.append(i + 2); indexes.append(i + 1); indexes.append(i + 3);
    }

    m_Objects_Cube.append(new SimpleObject3D(vertexes, indexes, img));
}

void Cube::add_cube(QVector3D pos)
{
  initCube(0.5f, *cube);

  m_Objects_Cube.back()->translate(pos);

  //qDebug()<< m_Objects_Cube.size();
}

void Cube::translation(QVector3D pos)
{
    if (m_Objects_Cube.isEmpty()) return; // Защита от пустых вызовов
      m_Objects_Cube.back()->translate(pos);
}

void Cube::Invisible_AllCube(bool invisible_)
{
   invisible = invisible_;
}

void Cube::Offset_AllCube(QVector3D pos)
{
  for(uint16_t i = 0; i < m_Objects_Cube.size(); i++) m_Objects_Cube[i]->offset(pos);
}

void Cube::delete_cube()
{
  m_Objects_Cube.clear();

}



