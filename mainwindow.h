#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "predefine.h"

class Fea;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_process_clicked();

//    void on_loadModel_clicked();

//    void on_bMatrixPath_clicked();

    void on_showImage_clicked();

    void on_load_clicked();

    void on_load2D_clicked();

    void on_load3D_clicked();

    void on_sightBall_clicked();

private:
    Ui::MainWindow *ui;
    Fea *fea;
};

#endif // MAINWINDOW_H
