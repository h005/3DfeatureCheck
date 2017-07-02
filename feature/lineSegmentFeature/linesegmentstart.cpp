//#include "linesegmentfea.h"

//void setLineSegmentFeature();

////int main()
////{
////}

//void setLineSegmentFeature()
//{
//    int NUM_Hist = 9;

//    double centroidRow = 0.0;
//    double centroidCol = 0.0;
//    setCentroid(centroidRow, centroidCol);

//    // line segment detection
//    LineSegmentFea *lsf = new LineSegmentFea();

//    double thLength = 30;
//    lsf->initial(image2D, thLength);
////    lsf->LineDetect( image2D, thLength);
//    // cam

//    std::vector<double> angleHist(NUM_Hist,0);
//    double variance = 0.0;
//    double entropy = 0.0;
////    double distance = 0.0;

//    lsf->setHist_v_e(angleHist, variance, entropy);

//    for(int i=0;i<NUM_Hist;i++)
//    {
//        fea2D.push_back(angleHist[i]);
//        fea2DName.push_back("LineSegment");
//    }

//    fea2D.push_back(variance);
//    fea2D.push_back(entropy);
//    fea2DName.push_back("LineSegment");
//    fea2DName.push_back("LineSegment");

//    double val_lb2ru = 0.0;
//    double val_lu2rb = 0.0;
//    lsf->setMinDiagonalAngle(val_lb2ru, val_lu2rb);

//    fea2D.push_back(val_lb2ru);
//    fea2D.push_back(val_lu2rb);

//    fea2DName.push_back("LineSegment");
//    fea2DName.push_back("LineSegment");
//}

