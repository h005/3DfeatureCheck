#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "standalone_image.h"
#include "gist.h"

#include <fstream>

#include <QStringList>
#include <QString>
#include <QFileInfo>

#define USERNAME "hejw005"

using namespace std;
using namespace cv;
using namespace cls;

const GISTParams DEFAULT_PARAMS {true, 32, 32, 4, 3, {8, 8, 4}};
//const GISTParams DEFAULT_PARAMS {true, 32, 32, 4, 3, {8, 8, 4}};

void setModelList(QStringList &modelList);

void setFileList(QString model, QStringList &fileList);

void setGistFeature(QString model, QStringList &fileList);

int main()
{
    QStringList modelList;
    setModelList(modelList);
    for(int i=0;i<modelList.size();i++)
    {
        QStringList fileList;
        setFileList(modelList.at(i), fileList);
        setGistFeature(modelList.at(i),fileList);
//        break;
    }

    return 0;
}

void setGistFeature(QString model, QStringList &fileList)
{

    QString gistFile = "/home/" + QString(USERNAME) + "/Documents/vpDataSet/tools/vpData/" + model + "/vpFea/" + model + ".gist960";
    std::fstream outGist;
    outGist.open(gistFile.toStdString(), std::fstream::out);
    Mat src;
    for(int i=0;i<fileList.size();i++)
    {
        src = imread(fileList.at(i).toStdString());
        if (src.empty()) {
            cerr << "No input image!" << endl;
            exit(1);
        }

        vector<float> result;
        GIST gist_ext(DEFAULT_PARAMS);
        gist_ext.extract(src, result);
        outGist << fileList.at(i).toStdString() << std::endl;
        for (const auto & val : result) {
            outGist << fixed << setprecision(4) << val << " ";
        }
        outGist << endl;

//        std::cout << fileList.at(i).toStdString() << std::endl;
        printf("%s\n",fileList.at(i).toStdString().c_str());
    }
    src.release();
    outGist.close();
}

void setModelList(QStringList &modelList)
{
    modelList.clear();
    modelList << "bigben"
              << "kxm"
              << "notredame"
              << "freeGodness"
              << "tajMahal"
              << "cctv3"
              << "BrandenburgGate"
              << "BritishMuseum"
              << "potalaPalace"
              << "capitol"
              << "Sacre"
              << "TengwangPavilion"
              << "mont"
              << "HelsinkiCathedral"
              << "BuckinghamPalace"
              << "castle"
              << "njuSample"
              << "njuSample2"
              << "njuSample3"
              << "njuActivity"
              << "njuActivity2";

//              << "model5"
    //                  << "house8"
    //                  << "pavilion9"
    //                  << "villa7s"

//    modelList.clear();
//    modelList << "njuSample3";
}

void setFileList(QString model, QStringList &fileList)
{

    fileList.clear();
    QString basePath = "/home/" + QString(USERNAME) + "/Documents/vpDataSet/";

    QString matrixFile = basePath + model + "/model/" + model + ".matrix";
    std::fstream finMatrix;
    finMatrix.open(matrixFile.toStdString(),std::fstream::in);
    string fileName;
    while(finMatrix >> fileName)
    {
        QFileInfo fileInfo(QString(fileName.c_str()));
        QString myFileName = basePath + model + "/imgs/" + fileInfo.fileName();
        fileList << myFileName;
        // it should be i < 8,
        // but there is still remains a '\n'
        // after finMatrix >> fileName,
        // so here should be i < 9
        for(int i=0;i<9;i++)
            std::getline(finMatrix,fileName);
    }
    finMatrix.close();


//    for(int i=0;i<fileList.size();i++)
//        std::cout << fileList.at(i).toStdString() << std::endl;

}
