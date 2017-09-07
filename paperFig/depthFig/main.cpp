#include <QString>
#include <QFile>
#include <QDir>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <opencv.hpp>
#include <vector>


void MyColorEnhance(QString src,QString dest)
{
    // read in
    cv::Mat img = cv::imread(src.toStdString().c_str(),0);
    cv::Mat dst(img.rows,img.cols,CV_8UC4,cv::Scalar(0,0,0,0));
    int treshold = 90;
    int hist[300];
    int chist[300];
    memset(hist,0,sizeof(int)*300);
    memset(chist,0,sizeof(int)*300);
    // show hist
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
            hist[img.at<uchar>(i,j)]++;
    }
    for(int i=0;i<256;i++)
        printf("index %d %d\n",i,hist[i]);
    printf("\n");

    std::vector<uchar> pixels;
    // remove all the pixels which the value larger than 240
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
            if(img.at<uchar>(i,j) <= treshold)
                pixels.push_back(img.at<uchar>(i,j));
    }
    cv::Mat tmpImg(1,pixels.size(),CV_8UC1);
    for(int i=0;i<pixels.size();i++)
        tmpImg.at<uchar>(0,i) = pixels[i];
    cv::equalizeHist(tmpImg,tmpImg);

    cv::Mat tmpDst;
//    printf("apply color map");
    cv::applyColorMap(tmpImg,tmpDst,cv::COLORMAP_JET);
    int index = 0;
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(img.at<uchar>(i,j) <= treshold)
            {
                dst.at<cv::Vec4b>(i,j)[0] = tmpDst.at<cv::Vec3b>(0,index)[0];
                dst.at<cv::Vec4b>(i,j)[1] = tmpDst.at<cv::Vec3b>(0,index)[1];
                dst.at<cv::Vec4b>(i,j)[2] = tmpDst.at<cv::Vec3b>(0,index++)[2];
                dst.at<cv::Vec4b>(i,j)[3] = 255;
//                printf("%d/%d %d %d %d\n",index,pixels.size(),tmpDst.at<cv::Vec3b>(1,index-1)[0],
//                        tmpDst.at<cv::Vec3b>(1,index-1)[1],
//                        tmpDst.at<cv::Vec3b>(1,index-1)[2]);
//                if(index == 989)
//                {
//                    printf("debug");
//                }
            }
        }
    }



//    cv::namedWindow("heatmap");
//    cv::imshow("heatmap",dst);
//    cv::waitKey(0);

    cv::imwrite(dest.toStdString().c_str(),dst);

}

void saveImg()
{
    QString imgFile = "/home/h005/Downloads/depth.txt";
    QString imgFileOut = "/home/h005/Downloads/depth2.txt";
    std::fstream fin;
    std::fstream fout;
    fin.open(imgFile.toStdString(),std::fstream::in);
    fout.open(imgFileOut.toStdString(), std::fstream::out);

    int rows, cols;
    fin >> rows >> cols;
    fout << rows << " " << cols << std::endl;
    cv::Mat img;
    int val;
    img.create(rows,cols,CV_16UC1);
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            fin >> val;
            fout << val << " ";
            img.at<ushort>(i,j) = val;
        }
        fout << std::endl;
    }



    cv::namedWindow("img");
    cv::imshow("img",img);
    cv::waitKey(0);

    fin.close();
    fout.close();

    cv::imwrite("/home/h005/Downloads/depth.png",img);
    std::cout << "done" << std::endl;
}

int main()
{
    QString file = "/home/h005/Documents/vpDataSet/kxm/model/depth/img0826.jpg.jpg";
    QString dest = "/home/h005/Documents/vpDataSet/kxm/model/depth/img0826depth.png";
    MyColorEnhance(file,dest);

//    saveImg();
}
