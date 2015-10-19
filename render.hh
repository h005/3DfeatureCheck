#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include "meshglhelper.hh"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <glm/glm.hpp>
#include "common.hh"
#include "externalimporter.hh"
#include <QString>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)

class PointsMatchRelation;

class Render : public QOpenGLWidget,protected QOpenGLFunctions
{
    Q_OBJECT

public:
    Render(MyMesh &in_mesh, QString fileName, QWidget *parent = 0);

    Render(MyMesh &in_mesh, glm::mat4 &model, glm::mat4 &view, glm::mat4 &projection, QWidget *parent = 0);

    void setMVP(glm::mat4 &model,glm::mat4 &view,glm::mat4 &proj);

    void setParameters();

    void setMeshSaliencyPara(ExternalImporter<MyMesh> *exImporter);

    void clear();

    void initial();

    void resizeGL(int width,int height);

    QSize sizeHint() const;

    QSize minimumSizeHint() const;

    bool rendering(int count);

    void showImage();
    // fileName is absolute name
    void storeImage(QString path,QString fileName);

    ~Render();



public slots:
    virtual void cleanup();

signals:

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;


public:
    float *p_img;
    int p_width,p_height;
    std::vector<GLfloat> p_vertices;
    std::vector<bool> p_isVertexVisible;
    std::vector<GLuint> p_VisibleFaces;
    std::vector<GLfloat> p_verticesMvp;
    glm::mat4 p_model;
    std::vector<MyMesh> p_vecMesh;
    std::vector<std::vector<int>> p_indiceArray;

protected:
    glm::mat4 getModelViewMatrix();
    glm::mat4 getModelMatrix();
    glm::mat4 m_model;
    glm::mat4 m_view;
    glm::mat4 m_proj;
/*
    glm::mat4 m_camera;
    glm::mat4 m_baseRotate;
    glm::vec3 m_rotateN;
    GLfloat m_angle;
    GLfloat m_scale;
*/
    GLuint m_programID;
    MeshGLHelper<MyMesh> m_helper;
    MyMesh m_mesh;

private:
    QString fileName;
    bool m_transparent;
    GLuint frameBufferId;
    GLuint depthRenderBuffer;
    GLuint colorRenderBuffer;

};

#endif // RENDER_H
