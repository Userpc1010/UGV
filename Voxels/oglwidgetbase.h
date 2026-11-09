#ifndef OGLWIDGETBASE_H
#define OGLWIDGETBASE_H

#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_4_5_Core>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QQuaternion>
#include <QVector3D>
#include <QQuaternion>
#include <Eigen/Core>
#include <Eigen/Geometry>


#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <Eigen/Dense>
#include <cmath>

struct DlioFrame
{
    // ===================================================================
    //  БЛОК НАСТРОЕК (КОНФИГ) — Можно менять в рантайме или при старте
    // ===================================================================

    // 1) Базовая перестановка и знаки осей (DLIO -> GL базис)
    inline static Eigen::Matrix3f R_perm =
        (Eigen::Matrix3f() <<
             0, -1,  0,
             0,  0,  1,
            -1,  0,  0).finished();

    // 2) Настройки фазового доворота (Yaw Fix)
    inline static float yaw_fix_angle = -M_PI / 2.0f;          // Угол в радианах (-90°)
    inline static Eigen::Vector3f yaw_fix_axis = Eigen::Vector3f::UnitY(); // Ось вращения в GL (UnitY)

    // ===================================================================
    //  ВНУТРЕННИЙ КЛЮЧЕВОЙ МЕТОД: Генерация динамической матрицы доворота
    // ===================================================================
    static Eigen::Matrix3f get_yawfix_matrix() {
        return Eigen::AngleAxisf(yaw_fix_angle, yaw_fix_axis).toRotationMatrix();
    }

    // ===================================================================
    //  ВНЕШНИЙ ИНТЕРФЕЙС: Принимают исходные данные и отдают ГОТОВЫЙ результат
    // ===================================================================

    // ── МАТРИЦА 3×3: Вход M_in (одометрия) ──> Выход (готовая для шейдера)
    static Eigen::Matrix3f rot(const Eigen::Matrix3f& M_in) {
        // Шаг 1: Перестановка базиса через сопряжение
        Eigen::Matrix3f R_odom = R_perm * M_in * R_perm.transpose();

        // Шаг 2: Левое умножение на динамический доворот и возврат
        return get_yawfix_matrix() * R_odom;
    }

    // ── ВЕКТОР / ПОЗИЦИЯ: Точно такой же двухэтапный конвейер для векторов ──
    static Eigen::Vector3f vec(const Eigen::Vector3f& v_in) {
        // Шаг 1: Перевод вектора в новый базис осей
        Eigen::Vector3f v_odom = R_perm * v_in;

        // Шаг 2: Левое умножение на доворот (чтобы ровер не ехал боком!)
        return get_yawfix_matrix() * v_odom;
    }

    // ── КВАТЕРНИОН: Автоматически использует настроенный метод rot ──
    static Eigen::Quaternionf quat(const Eigen::Quaternionf& q_in) {
        return Eigen::Quaternionf(rot(q_in.toRotationMatrix()));
    }

    // ── АФФИННАЯ МАТРИЦА 4×4: Полный сборщик Pose одной функцией ──
    static Eigen::Matrix4f mat4(const Eigen::Matrix4f& T_in) {
        Eigen::Matrix4f T_gl = Eigen::Matrix4f::Identity();
        T_gl.block<3,3>(0,0) = rot(T_in.block<3,3>(0,0));
        T_gl.block<3,1>(0,3) = vec(T_in.block<3,1>(0,3));
        return T_gl;
    }

    // Обратные функции (если понадобятся для обратного пересчета)
    static Eigen::Matrix3f rot_inv(const Eigen::Matrix3f& M_gl) {
        Eigen::Matrix3f R_odom = get_yawfix_matrix().transpose() * M_gl;
        return R_perm.transpose() * R_odom * R_perm;
    }
    static Eigen::Vector3f vec_inv(const Eigen::Vector3f& v_gl) {
        return R_perm.transpose() * (get_yawfix_matrix().transpose() * v_gl);
    }
};


class SimpleObject3D;
class Camera_3D;
struct VertexData;

class OGLWidgetBase : public QOpenGLWidget
{
    Q_OBJECT

public:
    explicit OGLWidgetBase(QWidget *parent = nullptr);
    virtual ~OGLWidgetBase();

protected:
    // Виртуальные методы событий - наследники могут переопределить
    virtual void onMousePress(QMouseEvent* event);
    virtual void onMouseRelease(QMouseEvent* event);
    virtual void onMouseMove(QMouseEvent* event);
    virtual void onWheel(QWheelEvent* event);
    virtual void onKeyPress(QKeyEvent* event);

signals:
    void RayCastPosition(QVector3D CameraPos, QVector2D CameraRot);
    void reset();

protected:
    // Абстрактные методы - должны быть реализованы в наследниках
    virtual void initializeSpecific() = 0;
    virtual void paintSpecific() = 0;

    // Общие методы OpenGL
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    // Общие обработчики событий
    void mousePressEvent(QMouseEvent* event) final;
    void mouseReleaseEvent(QMouseEvent *event) final;
    void mouseMoveEvent(QMouseEvent* event) final;
    void wheelEvent(QWheelEvent* event) final;
    void keyPressEvent(QKeyEvent* event) final;

    // Общие вспомогательные методы
    bool initObj(const QString &path, const QImage &img);
    QVector3D screenCoordsToPlaneCoords(const QVector2D &mouse_pose);

    void setMapOffset(Eigen::Vector3f offset);

protected:
    // Общие поля для всех наследников
    QOpenGLFunctions_4_5_Core *function;
    QMatrix4x4 m_ProjectionMatrix;
    QOpenGLShaderProgram m_Program;
    Camera_3D *m_camera;
    SimpleObject3D *m_Rover;

    QVector3D init_pos;

    // Для управления камерой
    double lastX = 0.0, lastY = 0.0;
    bool cameraMousePressed = false;

    // Масштаб ровера
    float m_RoverScale = 1.0f;

    Eigen::Vector3f m_mapOffset = Eigen::Vector3f::Zero();  // Текущее смещение карты

private:
    void initShaders();
};

#endif // OGLWIDGETBASE_H
