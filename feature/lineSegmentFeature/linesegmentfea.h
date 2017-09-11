#ifndef LINESEGMENTFEA_H
#define LINESEGMENTFEA_H

//extern "C"
//{
#include "lsd.h"
//};
#include "vpdetection.h"

using namespace std;
using namespace cv;

class LineSegmentFea
{
public:
    LineSegmentFea();
    ~LineSegmentFea();
    void LineDetect( cv::Mat image,
                     double thLength);
    // users should call the initial function before setHist and setDistance
    void initial(cv::Mat &image, double thLength);

    // set Hist variance and entropy
    void setHist_v_e(std::vector<double> &angleHist,
                     double &variance,
                     double &entropy);

    // set Hist of diagonal lb2ru
    void setHist_diagonal(std::vector<double> &angleHist, std::string direction);

    void setClusterSize(std::vector<double> &clusterSize);

    void setMinDiagonalAngle(double &val_lb2ru, double &val_lu2rb);

    // set distance
    void setDistance(double &distance);

    void drawClusters( cv::Mat &img);

    // set Vanish points
    void setVanishPoints(std::vector<cv::Point2d> &vp2D);

private:
    // get teh angel by asin function
    double getAngleYaxis(cv::Point2d &vec);

    // get the angle by acos function
    double getAngleXaxis(cv::Point2d &vec);

    double getAngleDiagonal(cv::Point2d &vec, std::string direction);

    double getDiagonalAngleVal(cv::Point2d & vec, std::string direction);

private:
    std::vector< std::vector<double> > lines;
    std::vector< std::vector<int> > clusters;

    std::vector<cv::Point2d> vp2D;

    int imgRows;
    int imgCols;

};

#endif // LINESEGMENTFEA_H
