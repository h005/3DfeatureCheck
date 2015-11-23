#ifndef GRAPHCUT_H
#define GRAPHCUT_H

#include <QString>
#include <opencv.hpp>
#include "graph.h"

extern "C" {
#include <vl/generic.h>
#include <vl/gmm.h>
}

#define PI 3.141592653589793

class GraphCut
{
public:
    cv::Mat foregroundMask;
    cv::Mat backgroundMask;

    GraphCut();
    ~GraphCut();

    void readIn(QString file,QString fileMask);

    void cut();

    void cutting();

    void thinnig();

    void GraphCut::thinningIteration(cv::Mat& im, int iter);

    void makeMask(int fromX,int fromY,int toX,int toY);

    void makeMask();

    void test();

    void erode(int size,int times);

    void dilate(int size,int times);

    void setForegroundMask();

    void setBackgroundMask();

private:
    cv::Mat image;
    cv::Mat mask;
    cv::Mat gray;


    double scale = 10000;
    double MAX_=100000.0;
    double eps = 1e-5;

    double beta;
    double alpha;

    cv::Vec3b maskRGB;

    int numClusters;

    int kernelSize,opTimes;

    double *means1;
    double *covariances1;
    double *priors1;

    double *means2;
    double *covariances2;
    double *priors2;

    double *trainData2;
    double *trainData1;

    int getVecdis(cv::Vec3b a,cv::Vec3b b);

    double computeGMM(double r,double g,double b);
    double computeGMM2(double r,double g,double b);



};




#endif // GRAPHCUT_H
