#include "spheregenerator.h"

SphereGenerator::SphereGenerator(QString fileName)
{
    this->fileName = fileName;
}

void SphereGenerator::genObj()
{

    QFileInfo fileInfo(fileName);
    if(!fileInfo.exists())
    {
        std::cout << "error: file does not exist" << std::endl;
        return;
    }
    QDir baseDir(fileInfo.absoluteDir());
    QSettings settings(fileInfo.absoluteFilePath(),QSettings::IniFormat);
    // read in width
    sX = settings.value("sample/numX").toInt();
    sZ = settings.value("sample/numZ").toInt();
    sMatrixFile = QDir::cleanPath(QDir(baseDir).filePath(settings.value("sample/matrixfile").toString()));
    sTexture = QDir::cleanPath(QDir(baseDir).filePath(settings.value("sample/texture").toString()));
    sObj = QDir::cleanPath(QDir(baseDir).filePath(settings.value("sample/outputobj").toString()));

    // initialize
    projList.clear();
    mvList.clear();
    centerList.clear();

    readMatrix(sMatrixFile);
    deComposeMV();

    output();
}

void SphereGenerator::output()
{
//    texture = cv::imread(sTexture.toStdString().c_str());
//    QDateTime currTime = QDateTime::currentDateTime();
//    long long timeStamp = currTime.toMSecsSinceEpoch();
    std::ofstream out(sObj.toStdString().c_str());
    // for usemtl
    QString keyWords = "wire_sphere";
    out << "# created by h005 SphereGenerator" << std::endl;
    out << std::endl;
    // mtl file
    QFileInfo objFile(sObj);
    QString mtlFile(objFile.absolutePath());
    mtlFile.append("/");
    mtlFile.append(objFile.baseName());
    mtlFile.append(".mtl");
    QFileInfo mtlFileInfo(mtlFile);

    out << "mtllib "<< mtlFileInfo.fileName().toStdString() <<std::endl;

    for(int i=0;i<centerList.size();i++)
        out << "v "<< centerList.at(i).x
                  << " " << centerList.at(i).y
                  << " " << centerList.at(i).z << std::endl;

    out << "# "<< centerList.size() << " vertices" << std::endl;

    float uStep = 1.0 / (float)(sX-1);
    float vStep = 1.0 / (float)(sZ-1);

    float u = 1.0f,v = 1.0f;
    for(int i=0;i<sX;i++)
    {
        v = 1.0f;
        for(int j=0;j<sZ;j++)
        {
            out << "vt " << u << " " << v << std::endl;
            v -= vStep;
            if( v < 0.f)
                v = 0.f;
        }
        u -= uStep;
        if(u < 0.f)
            u = 0.f;
    }

    out << "# "<< centerList.size() << " texture coords" << std::endl;

    out << "g " << "Sphere 001"<< std::endl;

    out << "usemtl " << keyWords.toStdString() << std::endl;

    out << "s 1" << std::endl;

    for(int i=0;i<sX-1;i++)
    {
        for(int j=1;j<sZ;j++)
        {
            out << "f " << i * sZ + j << "/" << i * sZ + j << " ";
            out << (i+1) * sZ + j << "/" << (i+1) * sZ + j << " ";
            out << (i+1) * sZ + j + 1 << "/" << (i+1) * sZ + j + 1 << " ";
            out << i * sZ + j + 1 << "/" << i * sZ + j + 1 << std::endl;
        }
    }

    out << "# "<<(sX-1)*(sZ-1)<<" polygons"<<std::endl;
    out.close();

    std::cout << "mtlFile "<<mtlFile.toStdString() << std::endl;
    out.open(mtlFile.toStdString().c_str());

    out << "# created by h005 SphereGenerator" << std::endl;
    out << "newmtl wire_plane" << std::endl;
    out << "	Ns 32"<< std::endl;
    out << "	d 1"<< std::endl;
    out << "	Tr 0" << std::endl;
    out << "	Tf 1 1 1"<< std::endl;
    out << "	illum 2" << std::endl;
    out << "	Ka 0.7765 0.8824 0.3412"<< std::endl;
    out << "	Kd 0.7765 0.8824 0.3412"<< std::endl;
    out << "	Ks 0.3500 0.3500 0.3500"<< std::endl;
    QFileInfo textureInfo(sTexture);
    out << "	map_Ka "<< textureInfo.fileName().toStdString() << std::endl;
    out << "	map_Kd "<< textureInfo.fileName().toStdString() << std::endl;

    out.close();
}

void SphereGenerator::readMatrix(QString path)
{
    freopen(path.toStdString().c_str(),"r",stdin);
    char tmpss[200];
    float tmpNum;
    while(scanf("%s",tmpss)!=EOF)
    {
        glm::mat4 mv,p;
        for(int i=0;i<16;i++)
        {
            scanf("%f",&tmpNum);
            mv[i%4][i/4] = tmpNum;
        }
        mvList.push_back(mv);
        for(int i=0;i<16;i++)
        {
            scanf("%f",&tmpNum);
            p[i%4][i/4] = tmpNum;
        }
        projList.push_back(p);
    }
    fclose(stdin);
}

void SphereGenerator::deComposeMV()
{
    for(int i=0;i<mvList.size();i++)
    {
        glm::vec3 t = glm::vec3(mvList[i][3]);
        glm::mat3 R = glm::mat3(mvList[i]);

        glm::vec3 eye = - glm::transpose(R) * t;
        glm::vec3 center = glm::normalize(glm::transpose(R) * glm::vec3(0.f,0.f,-1.f)) + eye;

        centerList.push_back(center);
    }
}

