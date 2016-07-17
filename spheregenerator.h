#ifndef SPHEREGENERATOR_H
#define SPHEREGENERATOR_H

#include <QFileInfo>
#include <QSettings>
#include <vector>
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <QDateTime>
#include <opencv.hpp>
#include <QFileDialog>

class SphereGenerator
{
public:
    SphereGenerator(QString fileName);
    void genObj();
    void output();

private:
    void readMatrix(QString path);
    void deComposeMV();

private:
    QString fileName;
    std::vector<glm::mat4> projList;
    std::vector<glm::mat4> mvList;
    std::vector<glm::vec3> centerList;
    int sX;
    int sZ;
    QString sMatrixFile;
    QString sTexture;
    QString sObj;
    cv::Mat texture;
};

#endif // SPHEREGENERATOR_H
