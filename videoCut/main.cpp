#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <QString>
#include <QDir>
#include <QFileInfo>

int Time2mse(int h,int m,int s)
{
    int res = s;
    res = res + m * 60 + h * 60 * 60;
    return res*1000;
}
///
/// \brief cutToVideo
/// \param capture
/// \param outputFile
/// \param position
/// \param toPos
///
void cutToVideo(cv::VideoCapture capture, QString outputFile,double position, double toPos)
{
    cv::Mat frame;
    capture.read(frame);
    cv::VideoWriter output(outputFile.toStdString().c_str(),CV_FOURCC('M','J','P','G'),45,frame.size(),1);

    while(position <= toPos)
    {
        if(!capture.read(frame))
        {
            std::cout << "read failed" << std::endl;
            break;
        }
        output << frame;
        position++;
        std::cout << (long long)position << "/" << (long long)toPos << std::endl;
    }
}
///
/// \brief cutToImage
/// \param dir
/// \param capture
/// \param outputFile
/// \param position
/// \param toPos
/// \param step
///
void cutToImage(QDir dir,cv::VideoCapture capture, QString outputFile, double position, double toPos, int step)
{
    cv::Mat frame;
    capture.read(frame);
//    cv::VideoWriter output(outputFile.toStdString().c_str(),CV_FOURCC('M','J','P','G'),45,frame.size(),1);

    int index = 0;
    int count = 0;
    while(position <= toPos)
    {
        if(!capture.read(frame))
        {
            std::cout << "read failed" << std::endl;
            break;
        }
        if(count % step == 0)
        {
            QString num;
            num.setNum(index);
            cv::imwrite((dir.absolutePath()+"/img"+num+".jpg").toStdString().c_str(),frame);
            position+= step;
            index++;
        }
        count++;
        std::cout << (long long)position << "/" << (long long)toPos << std::endl;
    }
}

int main()
{
    QFileInfo fileInfo("/media/h005/083c1e3b-c763-4087-a08c-204937a2f57b/h005/Documents/taiwan1.mp4");
    QDir dir = fileInfo.absoluteDir();
    dir.mkdir(fileInfo.baseName() + "Cut");
    dir.cd(fileInfo.baseName() + "Cut");
    QString file("/media/h005/083c1e3b-c763-4087-a08c-204937a2f57b/h005/Documents/taiwan1.mp4");
    QString outputFile("/media/h005/083c1e3b-c763-4087-a08c-204937a2f57b/h005/Documents/taiwan1Cut.mp4");
    // open
    std::cout << file.toStdString() << std::endl;
    cv::VideoCapture capture(file.toStdString().c_str());
    std::cout << "readin ...." << std::endl;
    if(!capture.isOpened())
    {
        std::cout << "open failed";
        return 0;
    }
    std::cout << "readin done...." << std::endl;
    int h_from = 0;
    int m_from = 4;
    int s_from = 6;
    // cut from
    double position = Time2mse(h_from,m_from,s_from);
    capture.set(CV_CAP_PROP_POS_MSEC,position);
    double frameRate = capture.get(CV_CAP_PROP_FPS);
    int h_to = 0;
    int m_to = 4;
    int s_to = 9;
    double toPos = Time2mse(h_to,m_to,s_to);

    toPos = (toPos/1000 - position/1000)*frameRate + position;
    std::cout << "from " << position << std::endl;
    std::cout << "to " << toPos << std::endl;

    cutToVideo(capture,outputFile,position,toPos);

    capture.set(CV_CAP_PROP_POS_MSEC,position);
    double step = 3;
    cutToImage(dir,capture,outputFile,position,toPos,step);

//    cv::Mat frame;
//    capture.read(frame);
//    cv::VideoWriter output(outputFile.toStdString().c_str(),CV_FOURCC('M','J','P','G'),45,frame.size(),1);

//    while(position <= toPos)
//    {
//        if(!capture.read(frame))
//        {
//            std::cout << "read failed" << std::endl;
//            break;
//        }
//        output << frame;
//        position++;
//        std::cout << (long long)position << "/" << (long long)toPos << std::endl;
//    }
    std::cout <<"frameRate " << frameRate << std::endl;
    std::cout << "done" << std::endl;
}
