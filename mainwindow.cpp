#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "spheregenerator.h"
#include "fea.hh"
#include <QSettings>

#include <QFileDialog>
#include <QFileInfo>

#include <stdio.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    fea = NULL;
    ui->load->setShortcut(Qt::Key_L);
    ui->process->setShortcut(Qt::Key_P);
    ui->showImage->setShortcut(Qt::Key_S);

    char paraIn[50];
    scanf("%s",paraIn);
    QString file = "/home/h005/Documents/vpDataSet/";
    QString fileName = file.append(QString(paraIn));
    fileName.append("/model/");
    fileName.append(QString(paraIn));
    fileName = fileName.append(".obj");
    QFileInfo info(fileName);

    if(!strcmp(paraIn,"select"))
    {

    }
    else
    {
        while(!info.exists())
        {
            if(strcmp(paraIn,"exit")==0)
                return;
            std::cout << "model path error: model " << fileName.toStdString() << " doesn't exists "<< std::endl;
            scanf("%s",paraIn);
            file = "/home/h005/Documents/vpDataSet/";
            fileName = file.append(QString(paraIn));
            fileName.append("/model/");
            fileName.append(QString(paraIn));
            fileName = fileName.append(".obj");
            info.setFile(fileName);
        }

        ui->modelPath->setText(fileName);

        QFileInfo fileInfo(fileName);
        QString path = fileInfo.absoluteDir().absolutePath().append("/");

        fea = new Fea(fileName,path);
        // mode 0 means compute 2D and 3D freatuers together
        fea->setFeature(0);
    }

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_process_clicked()
{

}



void MainWindow::on_showImage_clicked()
{
    fea->showImage();
}

void MainWindow::on_load_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("matrix Files(*.off *.dae *.obj)"));
    if(fileName == NULL)
        return;

//    QString fileName = "/home/h005/Documents/kxm/debug/kxm.obj";

    ui->modelPath->setText(fileName);

    QFileInfo fileInfo(fileName);
    QString path = fileInfo.absoluteDir().absolutePath().append("/");

    fea = new Fea(fileName,path);
    // mode 0 means compute 2D and 3D freatuers together
    fea->setFeature(0);
}

void MainWindow::on_load2D_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("matrix Files(*.off *.dae *.obj)"));
    if(fileName == NULL)
        return;

    ui->modelPath->setText(fileName);

    QFileInfo fileInfo(fileName);
    QString path = fileInfo.absoluteDir().absolutePath().append("/");
//    qDebug() << path << endl;
    fea = new Fea(fileName,path);
    // mode 2 means compute 2D feature
    fea->setFeature(2);
}

void MainWindow::on_load3D_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("matrix Files(*.off *.dae *.obj)"));
    if(fileName == NULL)
        return;

    ui->modelPath->setText(fileName);

    QFileInfo fileInfo(fileName);
    QString path = fileInfo.absoluteDir().absolutePath().append("/");

    fea = new Fea(fileName,path);
    // mode 3  means computes 3D feature
    fea->setFeature(3);
}

void MainWindow::on_sightBall_clicked()
{
    // this function was used to export the matrix file on the sight ball
//    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"),
//                               ".matrix",
//                               tr("matrix (*.matrix)"));

    QString fileName = "/home/h005/Documents/vpDataSet/villa7s/model/villa7s.matrix";

    fea = new Fea();

    fea->exportSBM(fileName);

}

void MainWindow::on_vpSample_clicked()
{

    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("config Files(*.ini)"));
//    QString fileName = "/home/h005/Documents/vpDataSet/njuGuLou/vpSample/configSample.ini";
    QFileInfo fileInfo(fileName);
//    if(!fileInfo.exists())
//    {
//        std::cout << "error: file does not exist" << std::endl;
//        return;
//    }

    QDir baseDir(fileInfo.absoluteDir());
    QSettings settings(fileInfo.absoluteFilePath(),QSettings::IniFormat);
    // 读取模型路径
    QString v_modelPath = QDir::cleanPath(QDir(baseDir).filePath(settings.value("model/path").toString()));
    // matrix file路径
    QString v_matrixPath = QDir::cleanPath(QDir(baseDir).filePath(settings.value("model/matrix").toString()));
    // sampleIndex
    int sampleIndex = settings.value("model/sampleIndex").toInt();
    int numSamples = 50; // maximum num of sample points
    QString v_outputFile = QDir::cleanPath(QDir(baseDir).filePath(settings.value("model/output").toString()));

    QFileInfo modelFileInfo(v_modelPath);
    QString path = modelFileInfo.absoluteDir().absolutePath().append("/");

    std::cout << v_modelPath.toStdString() << std::endl;
    std::cout << path.toStdString() << std::endl;
    fea = new Fea(v_modelPath, path);
    std::cout << "load done" << std::endl;
    fea->viewpointSample(v_matrixPath,sampleIndex,numSamples,v_outputFile,fileInfo.absolutePath());
}
///
/// \brief MainWindow::on_sphereGen_clicked
/// generate sphere object with texture
///
void MainWindow::on_sphereGen_clicked()
{
        QString fileName = QFileDialog::getOpenFileName(this,
                          tr("Open"),".",
                          tr("config Files(*.ini)"));

        QFileInfo fileInfo(fileName);
        if(!fileInfo.exists())
        {
            std::cout << "error: file does not exist" << std::endl;
            return;
        }

        SphereGenerator *sGen = new SphereGenerator(fileName);
        sGen->genObj();
        qDebug()<< "generate done" << endl;
}
