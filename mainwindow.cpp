﻿#include "mainwindow.h"
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
//    ui->loadModel->setShortcut(Qt::Key_L);
//    ui->bMatrixPath->setShortcut(Qt::Key_M);
    ui->load->setShortcut(Qt::Key_L);
    ui->process->setShortcut(Qt::Key_P);
    ui->showImage->setShortcut(Qt::Key_S);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_process_clicked()
{
//    fea->setFeature();
}


void MainWindow::on_showImage_clicked()
{
    fea->showImage();
}

void MainWindow::on_load_clicked()
{
//    QString fileName = QFileDialog::getOpenFileName(this,
//                      tr("Open"),".",
//                      tr("matrix Files(*.off *.dae *.obj)"));
//    if(fileName == NULL)
//        return;

    QString fileName = "/home/h005/Documents/kxm/debug/kxm.obj";

    ui->modelPath->setText(fileName);

    QFileInfo fileInfo(fileName);
    QString path = fileInfo.absoluteDir().absolutePath().append("/");
//    qDebug() << path << endl;
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
//    qDebug() << path << endl;
    fea = new Fea(fileName,path);
    // mode 3  means computes 3D feature
    fea->setFeature(3);
}
