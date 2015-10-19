#ifndef FEA_H
#define FEA_H

#include <QString>
#include <QDir>
#include <QStringList>
#include <opencv.hpp>
#include <GL/glew.h>
#include "common.hh"
#include "render.hh"
#include "externalimporter.hh"
#include <iostream>
#include "meancurvature.hh"

#define NumHistDepth 15
#define NumHistViewEntropy 15
#define PI 3.1415926
#define MAX_LEN 128


class Fea
{
private:

    int t_case;
    int bound;

    QString path;
    QString output;
    QString matrixPath;

    QStringList fileName;
    QStringList pFileName;

    glm::mat4 m_model;
    glm::mat4 m_view;
    glm::mat4 m_projection;
    glm::mat4 m_abv;

//    std::vector<glm::mat4> model;
//    std::vector<glm::mat4> view;
//    std::vector<glm::mat4> projection;

    int NUM;
    int P_NUM;

    cv::Mat image;
    CvSeq *contour;

    MyMesh mesh;

    ExternalImporter<MyMesh> *exImporter;

    Render *render;

    double *feaArray;

public:
    void showImage();

    Fea(QString modelFile, QString path);

    void setFeature();

    void setMatrixPara(QString matrixPath);
    ~Fea();

private:

    void setMat(float *img, int width, int height);

    void setProjectArea();

    void setVisSurfaceArea(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setViewpointEntropy(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setSilhouetteLength();

    void setSilhouetteCE();

    void setMaxDepth(float *array, int len);

    void setDepthDistribute(GLfloat *zBuffer, int num);

    void setMeanCurvature(MyMesh mesh, std::vector<bool> &isVertexVisible);

    void setMeanCurvature(MeanCurvature<MyMesh> &a, std::vector<bool> &isVertexVisible);

    void setMeanCurvature(int t_case, std::vector<bool> &isVertexVisible,
                          std::vector<MyMesh> &vecMesh,std::vector<std::vector<int>> &indiceArray);

    void setGaussianCurvature(MyMesh mesh, std::vector<bool> &isVertexVisible);

    void setGaussianCurvature(int t_case,std::vector<bool> &isVertexVisible,
                              std::vector<MyMesh> &vecMesh, std::vector<std::vector<int>> &indiceArray);

    void setMeshSaliency(int t_case, std::vector<GLfloat> &vertex, std::vector<bool> &isVertexVisible,
                         std::vector<MyMesh> &vecMesh, std::vector<std::vector<int>> &indiceArray);

    void setMeshSaliency(MyMesh mesh,std::vector<GLfloat> &vertex,std::vector<bool> &isVertexVisible);

    void setMeshSaliency(MeanCurvature<MyMesh> &a,std::vector<GLfloat> &vertex,std::vector<bool> &isVertexVisible);

    void setAbovePreference(double theta);

    void setAbovePreference(glm::mat4 &model2,glm::mat4 &modelView);

    double getMeshSaliencyLocalMax(double *nearDis,int len,std::vector<double> meshSaliency);

    double getGaussWeightedVal(double meanCur,double *nearDis,int len,double sigma);

    double getDiagonalLength(std::vector<GLfloat> &vertex);

    void setNearDisMeshSaliency(std::vector<GLfloat> &vertex,int i,double len,double sigma,double *neardis);

    void vertexBoundBox(double *v,std::vector<GLfloat> &vertex,int i,int label);

    bool getCurvature(CvPoint2D64f *a,CvPoint2D64f *b,CvPoint2D64f *c,double &cur);

    void readCvSeqTest(CvSeq *seq);

    double getArea2D(CvPoint2D64f *a,CvPoint2D64f *b,CvPoint2D64f *c);

    double getArea3D(CvPoint3D64f *a,CvPoint3D64f *b,CvPoint3D64f *c);

    double getDis3D(std::vector<float> &vertex,int i1,int i2);

    double getDis2D(CvPoint2D64f *a,CvPoint2D64f *b);
//    get cosi0i2i1
    double cosVal3D(std::vector<float> &vertex,int i0,int i1,int i2);
//    get cosACB
    double cosVal2D(CvPoint2D64f *a,CvPoint2D64f *b,CvPoint2D64f *c);

    bool getR(CvPoint2D64f *a,CvPoint2D64f *b,CvPoint2D64f *c,double &r);

    void normalizeHist(double *hist,double step,int num);

    void initial();

    void setFilenameList_mvpMatrix(QString matrixFile);

    void print(QString p_path);

    void printOut();

    void set_tCase();

    void computeModel(glm::mat4 &m_model_tmp);

};

#endif // FEA_H
