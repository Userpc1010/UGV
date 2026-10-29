#include "multidrawline.h"
MultiDrawLine::MultiDrawLine()
{}

void MultiDrawLine::draw_lines(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
  for(auto o: m_Objects_Line) o->draw(program, functions);
}

void MultiDrawLine::Add_Lines(GLfloat* vertices_buffer, QVector3D color_buffer, unsigned long long counter, uint16_t index)
{
  m_Objects_Line.append(new DrawLine());

  m_Objects_Line[index - 1]->DrawLines(vertices_buffer, color_buffer, counter);

  //  qDebug()<<"Add Lines: "<<m_Objects_Line.size();
}

void MultiDrawLine::Remove_Lines(uint16_t index)
{
  if(index) m_Objects_Line.remove(index - 1);
}

void MultiDrawLine::Clear_Lines()
{
 m_Objects_Line.clear();
// qDebug()<<"Clear Lines";
}

