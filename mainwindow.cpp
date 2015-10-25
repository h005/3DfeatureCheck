#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "fea.hh"

#include <QFileDialog>
#include <QFileInfo>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    fea = NULL;
    ui->loadModel->setShortcut(Qt::Key_L);
    ui->bMatrixPath->setShortcut(Qt::Key_M);
    ui->process->setShortcut(Qt::Key_P);
    ui->showImage->setShortcut(Qt::Key_S);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_process_clicked()
{
    fea->setFeature();
}


void MainWindow::on_loadModel_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("matrix Files(*.off)"));
    if(fileName == NULL)
        return;

    ui->modelPath->setText(fileName);

    QFileInfo fileInfo(fileName);
    QString path = fileInfo.absoluteDir().absolutePath().append("/");

    fea = new Fea(fileName,path);

//    fea->setModelPath(fileName);
}

void MainWindow::on_bMatrixPath_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                      tr("Open"),".",
                      tr("matrix Files(*.mm)"));
    if(fileName == NULL)
        return;
//    QFileInfo fileInfo(fileName);
//    QString path = fileInfo.absoluteDir().absolutePath().append("/");

//    fea = new Fea(fileName,path);
    fea->setMatrixPara(fileName);

    ui->lmatrixPath->setText(QString("path: ").append(fileName));

}

void MainWindow::on_showImage_clicked()
{
    fea->showImage();
}
