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

    QString fileName = "/home/h005/Documents/vpDataSet/villa7_1/model/villa.matrix";

    fea = new Fea();

    fea->exportSBM(fileName);

}
