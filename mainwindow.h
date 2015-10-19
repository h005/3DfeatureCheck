#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

    void on_loadModel_clicked();

    void on_bMatrixPath_clicked();

    void on_showImage_clicked();

private:
    Ui::MainWindow *ui;
    Fea *fea;
};

#endif // MAINWINDOW_H
