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
#include "gausscurvature.hh"

#define NumHistDepth 20000
#define NumHistViewEntropy 15
#define PI 3.1415926
#define MAX_LEN 32
#define NUM_Distribution 4096


class Fea
{
    static const int FEA_NUM = 13;
private:

    int t_case;
    int bound;

    QString path;
    QString output;
    // path for .matrix
    QString matrixPath;
    // path for .mm
    QString mmPath;

    QStringList fileName;
    QStringList pFileName;

    glm::mat4 m_model;
    glm::mat4 m_view;
    glm::mat4 m_projection;
    glm::mat4 m_abv;

    std::vector<glm::mat4> m_modelList;
    std::vector<glm::mat4> m_viewList;
    std::vector<glm::mat4> m_projectionList;

    int NUM;
    int P_NUM;

    // render image
    cv::Mat image;
    // mask is a gray image only have 255 and 0 two values
    // 255 means foreground
    cv::Mat mask;
    // 2Dimage
    cv::Mat image2D;

    // used for contour
//    CvMemStorage *mem_storage;
    std::vector<std::vector<cv::Point>> contour;

    MyMesh mesh;

    ExternalImporter<MyMesh> *exImporter;

    Render *render;

    double *feaArray;

    std::vector<double> fea2D;

public:
    void showImage();

    Fea(QString modelFile, QString path);

    void setFeature();

    void setMMPara(QString matrixPath);

    ~Fea();

private:

    void readMask();

    void setMat(float *img, int width, int height,int dstWidth,int dstHeight);

    void setProjectArea();

    void setVisSurfaceArea(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setViewpointEntropy2(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setViewpointEntropy(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setSilhouetteLength();

    void setSilhouetteCE();

    void setMaxDepth(float *array, int len);

    void setDepthDistribute(float *zBuffer, int num);

    void setMeanCurvature(MeanCurvature<MyMesh> &a, std::vector<bool> &isVertexVisible);

    void setMeanCurvature(std::vector<MeanCurvature<MyMesh>> &a,
                          std::vector<bool> &isVertexVisible,
                          std::vector<std::vector<int>> &indiceArray);

    void setGaussianCurvature(GaussCurvature<MyMesh> &b, std::vector<bool> &isVertexVisible);

    void setGaussianCurvature(std::vector<GaussCurvature<MyMesh>> &a,
                              std::vector<bool> &isVertexVisible,
                              std::vector<std::vector<int>> &indiceArray);

    void setMeshSaliency(int t_case,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible,
                         std::vector<MeanCurvature<MyMesh>> &a,
                         std::vector<std::vector<int>> &indiceArray);

    void setMeshSaliency(std::vector<MeanCurvature<MyMesh>> &a,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible);

    void setMeshSaliencyCompute(std::vector<MeanCurvature<MyMesh>> &a,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible,
                         std::vector<std::vector<int>> &indiceArray);

    void setAbovePreference(double theta);

    void setAbovePreference(glm::mat4 &modelZ,glm::mat4 &model,glm::mat4 &view);

    void setAbovePreference(glm::mat4 &modelZ,glm::mat4 &modelView);

    void setOutlierCount();

    /*  2D feature */
    // fill fea2D 0~4095
    void getColorDistribution();
    // 4096
    void getHueCount();
    // 4097
    void getBlur();
    //    4098
    void getContrast();
    // 4099
    void getBrightness();
    // rule of thirds
    void getRuleOfThird();
    // lighting feature
    void getLightingFeature();
    // hog
    void getHog();

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

    void setMvpPara(QString matrixFile);

    void print(QString p_path);

    void printOut();

    void computeModel(glm::mat4 &m_view_tmp,glm::mat4 &m_model_tmp);

    void computeModel(glm::mat4 &m_model_tmp);

    double getContourCurvature(const std::vector<cv::Point2d> &points, int target);

    void clear();

};

#endif // FEA_H
