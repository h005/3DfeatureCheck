#include <QCoreApplication>

#include <stdio.h>
#include "graphcut.h"

extern GraphCut* graphCutInstance;
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    GraphCut *gc = new GraphCut();
    graphCutInstance = gc;

    QString name = "img0005.jpg";
    QString file = "E:/viewpoint/kxm/"+name;
    QString mask = "E:/ViewPoint/kxm/20151106/proj/"+name;

    gc->readIn(file,mask);

//    gc->dilate(5,20);

//    gc->erode(5,25);

//    gc->setForegroundMask();
//    gc->setBackgroundMask();

//    gc->cut();


//    gc->makeMask(145,140,186,209);
//    gc->makeMask(158,297,441,396);

//    gc->makeMask(46,24,410,211);



//    gc->thinnig();

//    gc->cut();
    gc->cutting();
//    gc->test();

    VL_PRINT ("Hello world!\n") ;

    return a.exec();
}
