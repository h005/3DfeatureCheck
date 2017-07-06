#include "linesegmentfea.h"

LineSegmentFea::LineSegmentFea()
{
}

LineSegmentFea::~LineSegmentFea()
{
    // release lines memory
    for(int i=0;i<lines.size();i++)
        std::vector<double>().swap(lines[i]);
    std::vector< std::vector <double> >().swap(lines);
    // release clusters memory
    for(int i=0;i<clusters.size();i++)
        std::vector<int>().swap(clusters[i]);
    std::vector< std::vector< int> >().swap(clusters);
}

void LineSegmentFea::LineDetect(cv::Mat image, double thLength)
{
    cv::Mat grayImage;
    if ( image.channels() == 1 )
        grayImage = image;
    else
        cv::cvtColor(image, grayImage, CV_BGR2GRAY);

    image_double imageLSD = new_image_double( grayImage.cols, grayImage.rows );
    unsigned char* im_src = (unsigned char*) grayImage.data;

    int xsize = grayImage.cols;
    int ysize = grayImage.rows;
    for ( int y = 0; y < ysize; ++y )
    {
        for ( int x = 0; x < xsize; ++x )
        {
            imageLSD->data[y * xsize + x] = im_src[y * xsize + x];
        }
    }

    ntuple_list linesLSD = lsd( imageLSD );
    free_image_double( imageLSD );

    int nLines = linesLSD->size;
    int dim = linesLSD->dim;
    std::vector<double> lineTemp( 4 );
    for ( int i = 0; i < nLines; ++i )
    {
        double x1 = linesLSD->values[i * dim + 0];
        double y1 = linesLSD->values[i * dim + 1];
        double x2 = linesLSD->values[i * dim + 2];
        double y2 = linesLSD->values[i * dim + 3];

        double l = sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) );
        if ( l > thLength )
        {
            lineTemp[0] = x1;
            lineTemp[1] = y1;
            lineTemp[2] = x2;
            lineTemp[3] = y2;

            lines.push_back( lineTemp );
        }
    }

    free_ntuple_list(linesLSD);
}

///
/// \brief LineSegmentFea::setHist_v_e set angle direction hist variance and entropy
/// \param angleHist
/// \param variance
/// \param entropy
///
void LineSegmentFea::setHist_v_e(std::vector<double> &angleHist,
                                 double &variance,
                                 double &entropy)
{
    const double pi = acos(-1);
    int nBins = angleHist.size();
    double step = pi / (double)nBins;
    double countItem = 0.0;

    for ( int idx = 0; idx < lines.size(); idx++)
    {
        cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
        cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );

        cv::Point2d vec = cv::Point2d(pt_e.x - pt_s.x,
                                      pt_e.y - pt_s.y);

        // add the hist compute code here
        double angle = getAngleXaxis(vec);
        int index = (int)(angle / step);
        if(index == nBins)
            index--;
        assert(index < nBins);
        angleHist[index]++;
        countItem++;
    }

// these code for only count the line segments in clusters
/*
    for ( int i = 0; i < clusters.size(); ++i )
    {
        for ( int j = 0; j < clusters[i].size(); ++j )
        {
            int idx = clusters[i][j];

            cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
            cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );

            cv::Point2d vec = cv::Point2d(pt_e.x - pt_s.x,
                                          pt_e.y - pt_s.y);

            // add the hist compute code here
            double angle = getAngleXaxis(vec);
            int index = (int)(angle / step);
            if(index == nBins)
                index--;
            assert(index < nBins);
            angleHist[index]++;
            countItem++;
        }
    }
*/
//    assert(countItem);

    double mean = 0.0;
    // normalize hist
    if(countItem != 0.0)
    {
        for(int i=0; i < nBins ; i++)
        {
            mean += angleHist[i] * (step / 2.0 + i * step);
            angleHist[i] = angleHist[i] / countItem;
        }
        mean /= countItem;
    }
    else
    {
        for(int i=0; i < nBins; i++)
            angleHist[i] = 0.0;
        mean = 0.0;
    }

    entropy = 0.0;
    for(int i=0; i < nBins; i++)
        if(angleHist[i])
            entropy += angleHist[i] * log2(angleHist[i]);
    entropy = -entropy;

    variance = 0.0;
    for(int i=0; i < nBins; i++)
        variance += angleHist[i] \
                * ((step / 2.0 + i * step) - mean) \
                * ((step / 2.0 + i * step) - mean);

}

void LineSegmentFea::setClusterSize(std::vector<double> &clusterSize)
{
    clusterSize.clear();
    if(lines.size() == 0 || lines.size() == 1)
    {
        clusterSize.push_back(0);
        clusterSize.push_back(0);
        clusterSize.push_back(0);
        return;
    }

    clusterSize.push_back(clusters[0].size());
    clusterSize.push_back(clusters[1].size());
    clusterSize.push_back(clusters[2].size());
}


double LineSegmentFea::getAngleXaxis(Point2d &vec)
{
    Point2d vecX = cv::Point2d(1.0,0.0);
    double val = sqrt(vec.x * vec.x + vec.y * vec.y);
    if(val < 1e-10)
    {
        std::cout << "warning: line segment length appears zero" << std::endl;
        return 0.0;
    }
    double angle = vecX.dot(vec) / val;
    return acos(angle);
}

double LineSegmentFea::getDiagonalAngleVal(Point2d &vec, string direction)
{
    // I define the val as the sin(theta) / ( len(vec) / len(diaVec) )
    // the minor, the better

    Point2d diaVec;
    if(direction == "lb2ru")
    {
        diaVec.x = -imgRows;
        diaVec.y = imgCols;
    }
    else if(direction == "lu2rb")
    {
        diaVec.x = imgRows;
        diaVec.y = imgCols;
    }
    else
    {
        std::cout << "diagonal direction error" << std::endl;
        assert(0);
    }

    // val = abs( || vec || * || diaVec || * sin(theta) )
    double val = vec.x * diaVec.y - diaVec.x * vec.y;
    val = val < 0 ? -val : val;

    double lenVec = sqrt(vec.dot(vec));

    // val = sin(theta) / ( || vec || / || diaVec || )
    // val = sin(theta) * || diaVec || / || vec ||
    assert(lenVec);
    val = val / lenVec / lenVec;
    return val;

}


void LineSegmentFea::setMinDiagonalAngle(double &val_lb2ru, double &val_lu2rb)
{

    // for the diagonal line of from left bottom 2 right up
    val_lb2ru = 1e50;
    // for the diagonal line of from left up 2 right bottom
    val_lu2rb = 1e50;

    for(int i=0;i<lines.size();i++)
    {
        cv::Point pt_s = cv::Point( lines[i][0], lines[i][1] );
        cv::Point pt_e = cv::Point( lines[i][2], lines[i][3] );

        cv::Point2d vec = cv::Point2d(pt_e.x - pt_s.x,
                                      pt_e.y - pt_s.y);

        double val = getDiagonalAngleVal(vec, "lb2ru");
        val_lb2ru = val < val_lb2ru ? val : val_lb2ru;

        val = getDiagonalAngleVal(vec, "lu2rb");
        val_lu2rb = val < val_lu2rb ? val : val_lu2rb;

    }
}

void LineSegmentFea::setDistance(double &distance)
{

}

void LineSegmentFea::drawClusters(cv::Mat &img)
{
    int cols = img.cols;
    int rows = img.rows;

    //draw lines
    std::vector<cv::Scalar> lineColors( 3 );
    lineColors[0] = cv::Scalar( 0, 0, 255 );
    lineColors[1] = cv::Scalar( 0, 255, 0 );
    lineColors[2] = cv::Scalar( 255, 0, 0 );

    for ( int i=0; i<lines.size(); ++i )
    {
        int idx = i;
        cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1]);
        cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3]);
        cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

        cv::line( img, pt_s, pt_e, cv::Scalar(0,0,0), 2, CV_AA );
    }

    for ( int i = 0; i < clusters.size(); ++i )
    {
        for ( int j = 0; j < clusters[i].size(); ++j )
        {
            int idx = clusters[i][j];

            cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
            cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );
            cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

            cv::line( img, pt_s, pt_e, lineColors[i], 2, CV_AA );
        }
    }
}


void LineSegmentFea::initial(Mat &image,
                             double thLength)
{
    // LSD line segment detection
    LineDetect( image, thLength );

    // Camera internal parameters
    cv::Point2d pp(image.cols / 2, image.rows / 2);
    double f = 6.053 / 0.009;          // Focal length (in pixel)

    this->imgCols = image.cols;
    this->imgRows = image.rows;

    // Vanishing point detection
    std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
    // clusters Line segment clustering results of each vanishing point
    VPDetection detector;
    // there are some info in the run function
    if(lines.size() == 0 || lines.size() == 1)
        return;
    detector.run( lines, pp, f, vps, clusters );

//    drawClusters( image );
//    for(int i=0;i<vps.size();i++)
//        std::cout << "vps " << vps[i] << std::endl;
//    imshow("",image);
//    cv::waitKey( 0 );
}



