//extern "C"
//{
#include "lsd.h"
//};
#include "vpdetection.h"

using namespace std;
using namespace cv;


// LSD line segment detection
void LineDetect( cv::Mat image, double thLength, std::vector<std::vector<double> > &lines )
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

void drawClusters( cv::Mat &img, std::vector<std::vector<double> > &lines, std::vector<std::vector<int> > &clusters )
{
    int cols = img.cols;
    int rows = img.rows;

    //draw lines
    std::vector<cv::Scalar> lineColors( 3 );
    lineColors[0] = cv::Scalar( 0, 0, 0 );
    lineColors[1] = cv::Scalar( 0, 0, 0 );
    lineColors[2] = cv::Scalar( 0, 0, 0 );


//    for ( int i=0; i<lines.size(); ++i )
//    {
//        int idx = i;
//        cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1]);
//        cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3]);
//        cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

//        cv::line( img, pt_s, pt_e, cv::Scalar(0,0,0), 2, CV_AA );
//    }


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



int main()
{
<<<<<<< HEAD
//    string inPutImage = "/home/h005/Documents/QtProject/3DfeatureCheck/lineSegmentDetection/P1020171.jpg";
    string inPutImage = "/home/hejw005/Documents/vpForTvcg/Figures/feature/img0099.jpg";
//"/home/hejw005/Documents/vpDataSet/notredame/imgs/alexindigo_380947116.jpg
    std::cout << inPutImage << std::endl;
=======
    string inPutImage = "/home/h005/Documents/QtProject/3DfeatureCheck/lineSegmentDetection/P1020171.jpg";
//    string inPutImage = "/home/h005/Documents/vpDataSet/notredame/imgs/alexindigo_380947116.jpg";
>>>>>>> 15230808640d9de55afb5e2fe786603650e781d5

    cv::Mat image= cv::imread( inPutImage );
    if ( image.empty() )
    {
        printf( "Load image error : %s\n", inPutImage );
    }

    // LSD line segment detection
    double thLength = 50.0;
    std::vector<std::vector<double> > lines;
    LineDetect( image, thLength, lines );

    // Camera internal parameters
//    std::cout << image.rows << " " << image.cols << std::endl;
//    cv::Point2d pp( 307, 251 );        // Principle point (in pixel)
    cv::Point2d pp(image.cols / 2, image.rows / 2);
    double f = 6.053 / 0.009;          // Focal length (in pixel)

    // Vanishing point detection
    std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
    std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
    VPDetection detector;
    detector.run( lines, pp, f, vps, clusters );

    string maskImg = "/home/hejw005/Documents/vpForTvcg/Figures/feature/img0099_.jpg";
    cv::Mat maskImage = cv::imread(maskImg);

    drawClusters( maskImage, lines, clusters );
    for(int i=0;i<vps.size();i++)
        std::cout << "vps " << vps[i] << std::endl;
    imshow("",maskImage);
    cv::waitKey( 0 );
    string outputImg = "/home/hejw005/Documents/vpForTvcg/Figures/feature/img0099__.jpg";
    cv::imwrite(outputImg,maskImage);
}
