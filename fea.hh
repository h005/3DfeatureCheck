﻿#ifndef FEA_H
#define FEA_H

#include <QString>
#include <QDir>
#include <QStringList>
#include <opencv.hpp>
#include <GL/glew.h>
#include <QFileInfo>
#include "common.hh"
#include "render.hh"
#include "externalimporter.hh"
#include <iostream>
#include "meancurvature.hh"
#include "gausscurvature.hh"

#define NumHistDepth 20000
#define NumHistViewEntropy 15
#define PI 3.1415926
#define MAX_LEN 16
#define NUM_Distribution 512

#define GLCM_DIS 3
#define GLCM_CLASS 16
#define NUM_GLCM_FEATURES 4
#define NUM_GLCM_DIRECTONS 3

#define CoWidth 64
#define CoHeight 64

#define USERNAME "h005"

class Fea
{
//    static const int FEA_NUM = 13;
private:

    int t_case;
    int bound;

    /* model path, including matrix file, mm file and model file
    also used to generate stored image path

    such as ~/Documents/vpDataSet/kxm/model/

    depth image
    ~/Documents/vpDataSet/kxm/model/depth/
    rgb image
    ~/Documents/vpDataSet/kxm/model/rgb/
    proj image
    ~/Documents/vpDataSet/kxm/model/proj/
    mask image
    ~/Documents/vpDataSet/kxm/model/mask/
    */
    QString path;
    // output both 2d and 3d fea
    QString output;
    QString outputFeaName;
    QString output2dFeaName;
    QString output3dFeaName;
    QString output2D;
    QString output3D;
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

    cv::Mat pcaResult;

    std::vector<glm::mat4> m_modelList;
    std::vector<glm::mat4> m_viewList;
    std::vector<glm::mat4> m_projectionList;

    int NUM;
    int P_NUM;

    // render image the same as background, in general cases image is the same as mask, when manipulate
    // the foregorund pixels, suggest you use the mask.
    cv::Mat image;
    // 255 means background
    cv::Mat mask;
    // 2Dimage
    cv::Mat image2D;
    // 2D gray Imge
    cv::Mat gray;
    // 2D CV_32FC3 hsv image
    cv::Mat image2D32f3c;


    // used for contour
//    CvMemStorage *mem_storage;
    std::vector< std::vector<cv::Point> > contour;

    MyMesh mesh;

    ExternalImporter<MyMesh> *exImporter;

    Render *render;

    std::vector<double> fea3D;

    std::vector<double> fea2D;

    std::vector<std::string> fea3DName;

    std::vector<std::string> fea2DName;

public:
    void showImage();

    Fea();

    Fea(QString modelFile, QString path);

    void exportSBM(QString file);

    void exportSBM_featureCheckModelNet40(QString file);

    void viewpointSample(QString v_matrixPath, int sampleIndex, int numSamples, QString output, QString configPath);

    void setFeature(int mode);

    void setMMPara(QString matrixPath);

    // this function was created to compute the gist features in the modelList
    void setFeatureGist(QStringList &modelList);
    // this function was created to compute the line segment direction features in the modelList
    void setFeatureLsd(QStringList &modelList);
    // this function was created to compute the line segment direction and the vanish feature in the modelList
    void setFeatureLsdVanish(QStringList &modelList);
    // this function was created to compute the surface of region of interest in the modelList
    void setFeatureROI(int mode, QString model);

    ~Fea();

private:

    void readMask();

    void setMask();

    void setMat(float *img, int width, int height,int dstWidth,int dstHeight);

    void setProjectArea();

    void setVisSurfaceArea(std::vector<GLfloat> &vertex, std::vector<GLuint> &face, double totalArea);

    void setViewpointEntropy2(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setViewpointEntropy(std::vector<GLfloat> &vertex, std::vector<GLuint> &face);

    void setSilhouetteLength();

    void setSilhouetteCE();

    void setMaxDepth(float *array, int len);

    void setDepthDistribute(float *zBuffer, int num);

    void setMeanCurvature(MeanCurvature<MyMesh> &a, std::vector<bool> &isVertexVisible);

    void setMeanCurvature(std::vector< MeanCurvature<MyMesh>* > &a,
                          std::vector<bool> &isVertexVisible,
                          std::vector< std::vector<int> > &indiceArray);

    void setGaussianCurvature(GaussCurvature<MyMesh> &b, std::vector<bool> &isVertexVisible);

    void setGaussianCurvature(std::vector< GaussCurvature<MyMesh>* > &a,
                              std::vector<bool> &isVertexVisible,
                              std::vector< std::vector<int> > &indiceArray);

    void setMeshSaliency(int t_case,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible,
                         std::vector< MeanCurvature<MyMesh> > &a,
                         std::vector< std::vector<int> > &indiceArray);

    void setMeshSaliency(std::vector< MeanCurvature<MyMesh> > &a,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible);

    void setMeshSaliencyCompute(std::vector< MeanCurvature<MyMesh> > &a,
                         std::vector<GLfloat> &vertex,
                         std::vector<bool> &isVertexVisible,
                         std::vector< std::vector<int> > &indiceArray);

    double setAbovePreference(double theta);

    void setAbovePreference(glm::mat4 &modelZ,glm::mat4 &model,glm::mat4 &view);

    void setAbovePreference(glm::mat4 &modelZ,glm::mat4 &modelView);

    void setOutlierCount();

    void setBoundingBox3D();

    void setBoundingBox3DAbs();

    void setTiltAngle(glm::mat4 &modelView);

    // reviewer 1, similar to outlier count
    // the area in the image to the whole image
    void setAreaRatio(glm::mat4 &projMatrix);

    void setRoi(double &roiVal, glm::mat4 modelView, std::vector<float> &v_roi, std::vector<float> &normal_vertex);

    void decomposeProjMatrix(glm::mat4 &porjMatrix, glm::mat4 &enlargedProjMatrix);

    /*  2D feature */
    // fill fea2D 0~4095
    void setColorDistribution();
    // 4096
    void setHueCount();
    // 4097
    void setBlur();
    //  4098
    void setContrast();
    // 4099
    void setBrightness();
    // rule of thirds
    void setRuleOfThird();
    // rule of thirds without model
    void setRuleOfThirds_withoutPhotos();
    // lighting feature
    void setLightingFeature();
    // glcm
    void setGLCM();
    // saliecny
    void setSaliency();
    // pca
    // add pca to fea2D
    void setPCA();
    // hog
    void setHog();
    // compute PCA
    void computePCA();
    // 3D bounding box 2D x y axis theta
    void set2DTheta();
    // 3D bounding box 2D x y axis theta
    void set2DThetaAbs();
    // color variance
    void setColorEntropyVariance();
    // color info including (RGB vlaue mean, HSV values (C1 in HSV space),  Hue histogram (5 bins and entropy) and Satuation)
    // ref Geometric Context from a Single Image ICCV 2005
    void setColorInfo();
    // ball coordinate
    void setBallCoord();
    // subject brightness
    void setSubjuctBrightness();
    // line segment features
    void setLineSegmentFeature();


    void generateGistFeature(QString model, QStringList &fileList);

    void generateLineSegmentFeature(QString model, QStringList &fileList);

    void generateLSD_VanishLine(QString model, QStringList &fileList);
    // helper function
    void setFileList(QString model, QStringList &fileList);


    void setCentroid(double &centroidRow, double &centroidCol);

    void setCentroid(double &centroidRow, double &centroidCol, cv::Mat &mask);

    void roundingBox2D(int &up,int &bottom,int &left,int &right);

    // rounding box used for HOG foreground
    void roundingBox(cv::Mat &boxImage);


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

    void printOut(int mode);

    void printFeaName(int mode = 0);

    void computeModel(glm::mat4 &m_view_tmp,glm::mat4 &m_model_tmp);

    void computeModel(glm::mat4 &m_model_tmp);

    double getContourCurvature(const std::vector<cv::Point2d> &points, int target);

    cv::Mat grade16(cv::Mat gray);

    //    index means the direction of GLCM
    //    glcmMatrix means the glcm matrix
    //    glcm stores the result
    void setGLCMfeatures(double *glcm,int index,double glcmMatrix[][GLCM_CLASS]);

    void deComposeMV(std::vector<glm::vec3> &eye,
                     std::vector<glm::vec3> &center,
                     std::vector<glm::vec3> &up);

    void clear();

    // 通过调整相机与模型坐标系的距离，来使整个建筑物都落入视口范围内
    void vpSampleWholeArchitecture();
    // 通过调整相机与模型坐标系原点的距离，使建筑物包围盒所占面积达到整个图像的一定比例
    void vpSampleArchitectureSize();
    // 通过调整相机的朝向使得建筑物居中
    void vpSampleArchitectureCenter();
    // 准备一下需要的参数，比如平均的相机距离，相机的朝向等
    void vpSamplePrepare(QString v_matrixPath,
                         float &distance,
                         float &distanceStep,
                         int &width,
                         int &height);

    float floatAbs(float num);
    double doubleAbs(double num);
    double getAngle(cv::Point2d u, cv::Point2d v);

    glm::mat4 normalizedModelView(const glm::mat4 &mvMatrix);

};


#endif // FEA_H
