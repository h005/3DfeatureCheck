#include "fea.hh"
#include "meancurvature.hh"
#include "gausscurvature.hh"
#include <QtDebug>
#include <QSettings>
#include "predefine.h"


Fea::Fea(QString fileName, QString path)
{
    fea3DName.clear();
    fea2DName.clear();

    this->path = path;

    // read in model
    exImporter = new ExternalImporter<MyMesh>();

    if(!exImporter->read_mesh(mesh,fileName.toStdString().c_str()))
    {
        std::cerr << "Error: Cannot read mesh from "<<std::endl;
        return ;
    }

#ifdef OUTPUT_OFF
    QString offName = fileName;
    int tmpPos = offName.lastIndexOf('.');
    offName.remove(tmpPos,20);
//    qDebug()<<"offName "<<offName<<endl;
    exImporter->outputMesh(mesh,offName);
#endif

    // If the file did not provide vertex normals, then calculate them
    if (!mesh.has_vertex_normals())
    {
        std::cout << "we need calculate vertex normal first" << std::endl;
        // allocate memory for normals storage
        // we need face normals to update the vertex normals
        mesh.request_face_normals();
        mesh.request_vertex_normals();

        // let the mesh update the normals
        mesh.update_normals();
        // dispose the face normals, as we don't need them anymore
        mesh.release_face_normals();
    }

    // for check we only need .mm file
    // this file also is a log file, save the ord will processed
    QString mmFile = fileName;
    int index = mmFile.lastIndexOf('.');
    mmFile.replace(index,6,".mm");
    setMMPara(mmFile);

#ifndef CHECK
    // for compute we also need .matrix file
    QString matrixFile = fileName;
    index = matrixFile.lastIndexOf('.');
    matrixFile.replace(index,10,".matrix");
    setMvpPara(matrixFile);
#endif

    glm::mat4 tmpPara;
    render = new Render(mesh,tmpPara,tmpPara,tmpPara);

    render->resize(QSize(800,800));
    render->show();

}


/*
 * for some model is so big, the memory is not enough
 * so ... set mode
 * mode 0 compute all features
 * mode 2 compute 2D features
 * mode 3 compute 3D features
 */
void Fea::setFeature(int mode)
{
    int fileNameOutputFlag = 0;
#ifdef CHECK
    glm::mat4 m_model_tmp;

    glm::mat4 m_view_tmp;

    MeanCurvature<MyMesh> a(mesh);
    GaussCurvature<MyMesh> b(mesh);

#else
//     for compute
    // 3D feature
    std::vector< MeanCurvature<MyMesh>* > a;
    std::vector< GaussCurvature<MyMesh>* > b;
    if(mode  == 3 || mode  == 0)
    {
        render->setMeshSaliencyPara(exImporter);
        for(int i=0;i<render->p_vecMesh.size();i++)
        {
            MeanCurvature<MyMesh> *tmpMean = new MeanCurvature<MyMesh>(render->p_vecMesh[i]);
            GaussCurvature<MyMesh> *tmpGauss = new GaussCurvature<MyMesh>(render->p_vecMesh[i]);
            a.push_back(tmpMean);
            b.push_back(tmpGauss);
        }
    }
    render->setAreaAllFaces();
    std::cout << "AreaAllFaces " << render->areaAllFaces << std::endl;
    std::cout << "case : " << t_case << std::endl;
#ifndef FOREGROUND
    // 关于PCA这里还有问题，是否要将前景物体提取出来然后再进行PCA计算？目前没有这么做
    computePCA();
#endif

#endif // without CHECK

    for(; t_case < NUM ; t_case++)
    {
        // render
#ifdef CHECK

        computeModel(m_view_tmp,m_model_tmp);

        render->setMVP(m_model_tmp,m_view_tmp,m_projection);
#else

        render->setMVP(m_modelList[t_case],m_viewList[t_case],m_projectionList[t_case]);
#endif

        bool res = render->rendering(t_case);

        if(res)
        {

            render->setParameters();
            //used for check the image
            //  render->showImage();
            int width  = 0;
            int height = 0;

            // read image2D used for 3D model to save proj and depth image
            image2D = cv::imread(fileName.at(t_case).toStdString().c_str());
//            std::cout << fileName.at(t_case).toStdString() << std::endl;
            qDebug() << fileName.at(t_case) << endl;
            if(mode == 2 || mode  ==0)
            {
                cv::cvtColor(image2D,gray,CV_BGR2GRAY);

                image2D32f3c = cv::Mat(image2D.rows,image2D.cols,CV_32FC3,cv::Scalar(0,0,0));
                // it is useful to convert CV_8UC3 to CV_32FC3
                image2D.convertTo(image2D32f3c,CV_32FC3,1/255.0);
                cv::cvtColor(image2D32f3c,image2D32f3c,CV_BGR2HSV);
            }

            width = image2D.cols;
            height = image2D.rows;


#ifdef CHECK
//            render->storeImage(path,QString::number(t_case));
#else
            // store rgb and depth img
            // need a folder named depth and rgb
            if(mode == 3 || mode  == 0)
                render->storeImage(path,fileName.at(t_case),width,height);
#endif

            setMat(render->p_img,render->p_width,render->p_height,width,height);

            setMask();
            // 0
            setProjectArea();
            // 1
            setVisSurfaceArea(render->p_vertices,render->p_VisibleFaces,render->areaAllFaces);
            // 2
            setViewpointEntropy2(render->p_verticesMvp,render->p_VisibleFaces);
            // 3
            setSilhouetteLength();
            // 4 5
            setSilhouetteCE();
            // 6
            setMaxDepth(render->p_img,render->p_height*render->p_width);
            // 7
            setDepthDistribute(render->p_img,render->p_height*render->p_width);
#ifdef CHECK

            setMeanCurvature(a,render->p_isVertexVisible);

            setGaussianCurvature(b,render->p_isVertexVisible);

//            setMeshSaliency(a,render->p_vertices,render->p_isVertexVisible);

            setAbovePreference(m_abv,m_model_tmp,m_view_tmp);

#else

            if(mode == 0 ||  mode == 3)
            {
                // 8
                setMeanCurvature(a,render->p_isVertexVisible,render->p_indiceArray);

                // 9
                setGaussianCurvature(b,render->p_isVertexVisible,render->p_indiceArray);
            }

//            setMeshSaliencyCompute(a,render->p_vertices,render->p_isVertexVisible,render->p_indiceArray);

            setAbovePreference(m_abv,
                               render->m_model,
                               render->m_view);

#endif

            setOutlierCount();

            setBoundingBox3DAbs();

            /*
              add 2D fea function here
            */

            // 可以只计算mask对应部分的前景区域
            // getColorDistribution();
            // 可以只计算mask对应部分的前景区域 2Dfea
            if(mode == 2 || mode == 0)
            {
                getHueCount();
                // 不计算
//                getBlur();
                // 可以只计算mask对应部分的前景区域
                getContrast(); // 直方图中占98%区域的宽度
                // 可以只计算mask对应部分的前景区域
                getBrightness();
            }
            // 球面坐标系
            getBallCoord();

//#ifndef FOREGROUND

//            getLightingFeature();

//            // 不计算
//            setGLCM();

//            setSaliency();

//            setPCA();

//#endif
            if(mode == 2 || mode == 0)
            {
                getRuleOfThird();
                // 各种不同方向的梯度叠加
                getHog();

//                get2DTheta();

                get2DThetaAbs();

                getColorEntropyVariance();

                getColorInfo();

            }

            image2D.release();

            image.release();

            mask.release();

            gray.release();

            image2D32f3c.release();
        }
//        break;
        if(!fileNameOutputFlag)
        {
            printFeaName();
            fileNameOutputFlag = !fileNameOutputFlag;
        }

        printOut(mode);

        clear();



        qDebug() << "fea cases : "<< t_case << endl;

    }

    qDebug() << "done" << endl;

}

void Fea::setMMPara(QString mmFile)
{
    this->mmPath = mmFile;
    // mmFile = /home/h005/Documents/vpDataSet/test/model/cctv.mm
    QFileInfo outputFile(mmFile);
    // output = /home/h005/Documents/vpDataSet/test/model
    output = outputFile.absolutePath();
    QFileInfo tmpOutputFile(output);
    // output = /home/h005/Documents/vpDataSet/test
    output = tmpOutputFile.absolutePath();
    output.append("/vpFea/");
    QString baseName = outputFile.baseName();
    output.append(baseName);
    output2D = output;
    output3D = output;
    outputFeaName = output;
    output2dFeaName = output;
    output3dFeaName = output;
    output2D.append(".2df");
    output3D.append(".3df");
    output.append(".fea");
    outputFeaName.append(".fname");
    output2dFeaName.append(".2dfname");
    output3dFeaName.append(".3dfname");

    if(!freopen(mmPath.toStdString().c_str(),"r",stdin))
    {
        t_case = 0;
        return;
    }


#ifdef CHECK
    // for check there are two matrixs
    // the first one is the view matrix
    // the second one is the model matrix and also is the abv matrix
    // the last two para is from to

    float tmp;
    for(int i=0;i<16;i++)
    {
        scanf("%f",&tmp);
        m_view[i/4][i%4] = tmp;
    }

    for(int i=0;i<16;i++)
    {
        scanf("%f",&tmp);
        m_abv[i/4][i%4] = tmp;
    }

    scanf("%d %d",&t_case,&NUM);

//    m_model = glm::transpose(m_model);

//    m_view = glm::mat4(1.f);
    m_model = m_abv;

    m_projection = glm::perspective(glm::pi<float>() / 2, 1.f, 0.1f, 100.f);
#else
    /* for compute there is one matirx is abv matrix
    // and the last two para is from to
    // 该矩阵是对任何给定的一个模型，经过调整使其模型的Z轴与OpenGL坐标系下的Z轴重合之后所经历过的变换，只有旋转，没有平移和缩放
    // rearrange the model needless to read the parameters

    //    float tmp;
    //    for(int i=0;i<16;i++)
    //    {
    //        scanf("%f",&tmp);
    //        m_abv[i/4][i%4] = tmp;
    //    }
    */
    m_abv = glm::mat4(1.f);
    scanf("%d",&t_case);
#endif

    fclose(stdin);
}


Fea::~Fea()
{

}

void Fea::readMask()
{
    QString file = fileName.at(t_case);
    int pos = file.lastIndexOf('/');
    file = file.remove(pos,20);
    // add mask folder
    file = file + "/mask";
    QString tmp = fileName.at(t_case);
    tmp = tmp.remove(0,pos);

    file = file + tmp;
    // mask is a gray image only have 255 and 0
    // where 255 means foreground
    mask = cv::imread(file.toStdString(),0);

    qDebug()<<"mask file name ..."<<file<<endl;

    if(mask.rows == 0 || mask.cols == 0)
    {
        qDebug()<<"error : mask file does not exist ... "<<endl;
//        image2D.copyTo(mask);
        cv::cvtColor(image2D,mask,CV_BGR2GRAY);
    }
//    cv::namedWindow("mask");
//    cv::imshow("mask",mask);

}

void Fea::setMask()
{
    image.copyTo(mask);
}
///
/// \brief Fea::setMat
/// \param img
/// \param width
/// \param height
/// \param dstWidth
/// \param dstHeight
/// image 存储灰度图像，其中255是背景区域
///
void Fea::setMat(float *img, int width, int height,int dstWidth,int dstHeight)
{
//    image.release();
    float min = 1.0;
    for(int i=0;i<width*height;i++)
        min = min < img[i] ? min : img[i];

    cv::Mat image0 = cv::Mat(width,height,CV_32FC1,img);
    image0.convertTo(image,CV_8UC1,255.0/(1.0 - min),255.0 * min / (min - 1.0));
    // 二值化
    cv::threshold(image, image, 240, 255, CV_THRESH_BINARY);
    cv::flip(image,image,0);
    cv::resize(image,image,cv::Size(dstWidth,dstHeight));
    // release memory
    image0.release();
}

void Fea::setProjectArea()
{
//    cv::namedWindow("image");
//    cv::imshow("image",image);
//    cv::waitKey(0);

//    IplImage *saveImageDebug = new IplImage(image);
//    cvSaveImage("/home/h005/Documents/vpDataSet/njuSample/model/640.jpg",saveImageDebug);

//    cv::namedWindow("image2");
//    cv::imshow("image2",image);
//    cv::waitKey(0);

    double res = 0.0;
    cv::Mat img = cv::Mat(image.rows,image.cols,CV_8UC1);
    cv::Mat foreGround = cv::Mat(image.rows,image.cols,CV_8UC4,cv::Scalar(0,0,0,0));
//    qDebug()<<"set project area img .... "<<image.rows<<" "<<image.cols<<endl;
    if(image.channels()==3)
    {
        std::cout << "set project area channels 3" << std::endl;
        for(int i=0;i<image.rows;i++)
            for(int j=0;j<image.cols;j++)
                if(image.at<cv::Vec3b>(i,j)[0]!=255
                   || image.at<cv::Vec3b>(i,j)[1]!=255
                   || image.at<cv::Vec3b>(i,j)[2]!=255)
                res++;

    }
    else
    {
        for(int i=0;i<image.rows;i++)
            for(int j=0;j<image.cols;j++)
            {
                if(image.at<uchar>(i,j)!=255)
                {
                    res++;
                    img.at<uchar>(i,j) = 255;
                    foreGround.at<cv::Vec4b>(i,j)[0] = image2D.at<cv::Vec3b>(i,j)[0];
                    foreGround.at<cv::Vec4b>(i,j)[1] = image2D.at<cv::Vec3b>(i,j)[1];
                    foreGround.at<cv::Vec4b>(i,j)[2] = image2D.at<cv::Vec3b>(i,j)[2];
                    foreGround.at<cv::Vec4b>(i,j)[3] = 255;
                }
                else
                {
                    img.at<uchar>(i,j) = 0;
                }
            }

    }
//    qDebug()<<"set project area froeGround done .... " <<endl;
    QString fileName0 = fileName.at(t_case);

    int pos = fileName0.lastIndexOf('/');

    QString fileName = fileName0.remove(0,pos+1);



    QString projPath = path + QString("proj/").append(fileName);

//    cv::flip(img,img,0);

    IplImage *saveImage = new IplImage(img);
    cvSaveImage(projPath.toStdString().c_str(),saveImage);
//    cvSaveImage(projPath.toStdString().c_str(),&(IplImage(image)));
    delete saveImage;

    res /= image.cols*image.rows;
    fea3D.push_back(res);
    fea3DName.push_back("projectArea");
    img.release();
    std::cout<<"fea projectArea "<<res<<" fea3D size "<<fea3D.size()<<std::endl;

    QString maskPath = path + QString("mask/").append(fileName);
    pos = maskPath.lastIndexOf('.');
    maskPath.replace(pos+1,5,"png");

    std::cout << "foreGround path " << maskPath.toStdString() << std::endl;

    saveImage = new IplImage(foreGround);
    cvSaveImage(maskPath.toStdString().c_str(),saveImage);
//    cv::namedWindow("test");
//    cv::imshow("test",foreGround);
//    cv::waitKey(0);
    foreGround.release();

    img.release();

}

void Fea::setVisSurfaceArea(std::vector<GLfloat> &vertex,
                             std::vector<GLuint> &face, double totalArea)
{
    double res = 0.0;
    for(int i=0 ; i<face.size() ; i+=3)
    {
/*
//        area = a*b*sin(ACB)/2
//        the first vertex is A the edge is a(BC)
//        the second vertex is B the edge is b(AC)
//        the third vertex is C the edge is c(AB)
        double a = getDis3D(vertex,face[i+1],face[i+2]);
        double b = getDis3D(vertex,face[i],face[i+2]);
//        double c = getDis(vertex,face(i),face(i+1));
        double cosACB = cosVal3D(vertex,face[i],face[i+1],face[i+2]);
        double sinACB = sqrt(1.0-cosACB*cosACB);
        visSurfaceArea[t_case] += a*b*sinACB/2.0;
*/
        CvPoint3D64f p1 = cvPoint3D64f(vertex[3*face[i]],vertex[3*face[i]+1],vertex[3*face[i]+2]);
        CvPoint3D64f p2 = cvPoint3D64f(vertex[3*face[i+1]],vertex[3*face[i+1]+1],vertex[3*face[i+1]+2]);
        CvPoint3D64f p3 = cvPoint3D64f(vertex[3*face[i+2]],vertex[3*face[i+2]+1],vertex[3*face[i+2]+2]);
        res += getArea3D(&p1,&p2,&p3);
    }
    Q_ASSERT(totalArea);
    res = res / totalArea;

    fea3D.push_back(res);
    fea3DName.push_back("visSurfaceArea");
    std::cout<<"fea visSurfaceArea "<< res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
    // used for test
}

void Fea::setViewpointEntropy2(std::vector<GLfloat> &vertex, std::vector<GLuint> &face)
{
    double res = 0.0;
    double area = 0.0;
    double totalArea = image.cols*image.rows;
//    qDebug()<<"viewPointEntropy ... "<<totalArea<<endl;
    for(int i=0;i<face.size();i+=3)
    {
        CvPoint2D64f a = cvPoint2D64f(vertex[face[i]*3],vertex[face[i]*3+1]);
        CvPoint2D64f b = cvPoint2D64f(vertex[face[i+1]*3],vertex[face[i+1]*3+1]);
        CvPoint2D64f c = cvPoint2D64f(vertex[face[i+2]*3],vertex[face[i+2]*3+1]);
        area = getArea2D(&a,&b,&c);
//        qDebug()<<"viewPointEntropy ... "<<area<<endl;
        if(area)
            res += area/totalArea * log2(area/totalArea);
        else
            qDebug()<<"viewpoint Entropy "<<area<<endl;
    }
    // background
    if((totalArea - fea3D[0]) > 0)
        res += (totalArea - fea3D[0])/totalArea * log2((totalArea - fea3D[0])/totalArea);

    res = - res;


    fea3D.push_back(res);
    fea3DName.push_back("viewPointEntropy");
    std::cout<<"fea viewpointEntropy "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
}

void Fea::setViewpointEntropy(std::vector<GLfloat> &vertex, std::vector<GLuint> &face)
{
    double res = 0.0;
//    double hist[15];
    double *hist = new double[NumHistViewEntropy];
//    还有此处，写成 double hist[15]; memest(hist,0,sizeof(hist));也会出错！
    memset(hist,0,sizeof(double)*NumHistViewEntropy);
    double *area = new double[face.size()/3];
    double min = 1e10;
    double max = -1.0;
    res = 0.0;
//    setArea
//    qDebug()<<"face size"<<face.size()<<endl;
    if(face.size())
    {
        for(int i=0;i<face.size();i+=3)
        {
            CvPoint2D64f a = cvPoint2D64f(vertex[face[i]*3],vertex[face[i]*3+1]);
            CvPoint2D64f b = cvPoint2D64f(vertex[face[i+1]*3],vertex[face[i+1]*3+1]);
            CvPoint2D64f c = cvPoint2D64f(vertex[face[i+2]*3],vertex[face[i+2]*3+1]);
    //        double bc = getDis2D(&b,&c);
    //        double ac = getDis2D(&a,&c);
    //        double cosACB = cosVal2D(&a,&b,&c);
    //        double sinACB = sqrt(1.0 - cosACB*cosACB);
    //        area[i/3] = bc*ac*sinACB/2.0;
            area[i/3] = getArea2D(&a,&b,&c);
//            qDebug()<< "area... "<<i<<" "<<area[i/3]<<endl;
            min = min > area[i/3] ? area[i/3] : min;
            max = max > area[i/3] ? max : area[i/3];
        }
    //    setHist
        double step = (max-min)/(double)(NumHistViewEntropy);
        for(int i=0;i<face.size()/3;i++)
        {
    //        qDebug()<<(int)((area[i] - min)/step)<<endl;
            if(area[i] == max)
                hist[NumHistViewEntropy - 1]++;
            else
                hist[(int)((area[i] - min)/step)] ++;
        }
        normalizeHist(hist,step,NumHistViewEntropy);
    }


//    setEntropy
    for(int i=0;i<NumHistViewEntropy;i++)
        if(hist[i])
            res += hist[i]*log2(hist[i]);
//    NND绝对的未解之谜！加了下面一句话会报错！
    delete []hist;
    res = - res;


//    delete []hist;
    delete []area;
/*
    freopen("e:/matlab/vpe.txt","w",stdout);
    for(int i=0;i<vertex.size();i+=3)
        printf("%f %f %f\n",vertex[i],vertex[i+1],vertex[i+2]);
    fclose(stdout);
*/
    fea3D.push_back(res);
    fea3DName.push_back("viewPointEntropy");
    std::cout<<"fea viewpointEntropy "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
}

void Fea::setSilhouetteLength()
{
    double res = 0.0;
    // ref http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
    cv::Mat gray;

    // ref http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
    // 这个一定要二值化，图像本身就基本都是白色，直接提取轮廓是拿不到结果的
    cv::threshold( image, gray, 254, 255.0,cv::THRESH_BINARY_INV );

    //    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(gray,contour,hierarchy,CV_RETR_LIST  ,CV_CHAIN_APPROX_NONE );

//    qDebug() << "setSilhouette contour size "<<contour.size()<<endl;

    if(contour.size())
        for(int i=0;i<contour.size();i++)
        {
            res += cv::arcLength(contour[i],true);
        }
    else
        res = 0.0;

    fea3D.push_back(res);
    fea3DName.push_back("silhouetteLength");
    std::cout<<"fea silhouetteLength "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
    std::vector<cv::Vec4i>().swap(hierarchy);

//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
// see contour result
/*
    cv::Mat drawing = cv::Mat::zeros(gray.size(),CV_8UC3);
//    for(int i=0;i<contours.size();i++)
    for(int i=0 ; i<1 ; i++)
    {
        cv::Scalar color = cv::Scalar(255,255,255);
        cv::drawContours(drawing,contour,i,color,2,8,hierarchy,0,cv::Point());
    }

    cv::namedWindow("contours");
    cv::imshow("contours",drawing);
*/
}

void Fea::setSilhouetteCE()
{
    double res0 = 0.0;
    double res1 = 0.0;
    double curva = 0;
    double dis = 0.0;

//    example
//    abcdefghabcde
//     ^  ->  ^
//    abc -> bcd -> def
//    if(contour.size())

    for(int k = 0;k<contour.size();k++)
    {
        for(int i=0;i<contour[k].size();i++)
        {
            cv::Point a0 = contour[k][i];
            cv::Point b0 = contour[k][(i+1)%contour[k].size()];
            cv::Point c0 = contour[k][(i+2)%contour[k].size()];
            CvPoint2D64f a = cvPoint2D64f((double)a0.x,(double)a0.y);
            CvPoint2D64f b = cvPoint2D64f((double)b0.x,(double)b0.y);
            CvPoint2D64f c = cvPoint2D64f((double)c0.x,(double)c0.y);

            std::vector<cv::Point2d> points;
            points.push_back(cv::Point2d(a.x, a.y));
            points.push_back(cv::Point2d(b.x, b.y));
            points.push_back(cv::Point2d(c.x, c.y));


                dis = getDis2D(&a,&b) + getDis2D(&b,&c);

            double curvab = getContourCurvature(points,1);
            if (std::isnan(curvab)) {
    //            qDebug()<<a.x<<" "<<a.y<<endl;
    //            qDebug()<<b.x<<" "<<b.y<<endl;
    //            qDebug()<<c.x<<" "<<c.y<<endl;
    //            assert(0);
            }
            else
            {
                res0 += floatAbs(curvab);
                res1 += curvab*curvab;
            }

    //        qDebug()<<"curvature a"<<curva<<" "<<floatAbs(curvab)<< " "<<floatAbs(curvab) - floatAbs(curva)<<endl;
        }
    }


    fea3D.push_back(res0);
    fea3DName.push_back("silhouetteCurvature");
    fea3D.push_back(res1);
    fea3DName.push_back("silhouetteCurvatureExtrema");
    std::cout<<"fea silhouetteCurvature "<<res0<<" fea3D size "<<fea3D.size()-1<<std::endl;
    std::cout<<"fea silhouetteCurvatureExtrema "<<res1<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
}

void Fea::setMaxDepth(float *array,int len)
{
    double res = -1.0;
    for(int i=0;i<len;i++)
        if(array[i] < 1.0)
            res = res > array[i] ? res : array[i];

    fea3D.push_back(res);
    fea3DName.push_back("maxDepth");
    std::cout<<"fea maxDepth "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
}

void Fea::setDepthDistribute(float *zBuffer, int num)
{
    double res = 0.0;
    double min = 1.0;
    double max = -1.0;
    double *hist = new double[NumHistDepth];
    memset(hist,0,sizeof(double)*NumHistDepth);
    for(int i=0;i<num;i++)
    {
        min = min > zBuffer[i] ? zBuffer[i] : min;
        if(zBuffer[i] < 1.0)
            max = max < zBuffer[i] ? zBuffer[i] : max;
    }

    double step = (max - min)/(double)NumHistDepth;
//    qDebug()<<"depth ... "<<step<<endl;
    if(step > 0.0)
    {
        // explain for if else below
        // such as min = 0 and max = 15 then step = 1
        // so the hist is [0,1),[1,2),[2,3)...[14,15)
        // max was omit!
        for(int i=0;i<num;i++)
        {
            if(zBuffer[i]==max)
                hist[NumHistDepth - 1]++;
            else if(zBuffer[i] < 1.0) // 数组越界错误
                hist[(int)((zBuffer[i]-min)/step)]++;
        }
        // normalizeHist
        normalizeHist(hist,step,NumHistDepth);
    }

    std::cout<<"step "<<step<<std::endl;
    for(int i=0; i<NumHistDepth; i++)
        res += hist[i]*hist[i]*step;
    res = 1 - res;


    delete []hist;
    fea3D.push_back(res);
    fea3DName.push_back("depthDistribute");
    std::cout<<"fea depthDistriubute "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"depth distribute"<<endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
}

void Fea::setMeanCurvature(MeanCurvature<MyMesh> &a, std::vector<bool> &isVertexVisible)
{
    double res = 0.0;
    res = a.getMeanCurvature(isVertexVisible);
    if(fea3D[1])
        res /= fea3D[1];

    fea3D.push_back(res);
    fea3DName.push_back("meanCurvature");
    std::cout<<"fea meanCurvature "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
}

void Fea::setMeanCurvature(std::vector< MeanCurvature<MyMesh>* > &a,
                           std::vector<bool> &isVertexVisible,
                           std::vector< std::vector<int> > &indiceArray)
{
    double res = 0.0;
//    qDebug()<<"set Mean Curvature "<<a.size()<<endl;

    res = 0.0;
    for(int i=0;i<a.size();i++)
    {
        // 查看在哪个mesh上面crash掉了
//        std::cout<<"setMeahCurvature.... "<<i<<std::endl;
        // for debug 第136个mesh出错了，输出这个mesh的信息

//        if(i==0)
//        {
//            // 输入文件名即可，outputMesh函数会加上.off后缀名
//            QString tmpPath = this->path;
//            tmpPath.append('/');
//            tmpPath.append(fileName.at(t_case));
//            tmpPath.append(QString::number(i));
//            exImporter->outputMesh(vecMesh[i],tmpPath);
//        }

        std::vector<bool> isVerVis;
        std::set<int> verIndice;
        for(int j=0;j<indiceArray[i].size();j++)
            verIndice.insert(indiceArray[i][j]);

        std::set<int>::iterator it = verIndice.begin();
        for(;it!=verIndice.end();it++)
            isVerVis.push_back(isVertexVisible[*it]);

        if(i == 709)
        {
            std::cout << "for debug" << std::endl;
        }
        res += a[i]->getMeanCurvature(isVerVis);
//        std::cout << "debug index " << i << std::endl;
//        Q_ASSERT(!std::isnan(res));
    }

//    fclose(stdout);

//    qDebug()<<"fea meanCurvature res "<<res<<endl;
//    qDebug()<<"fea meanCurvature fea3D[8] "<<fea3D[8]<<endl;
    if(fea3D[1])
        res /= fea3D[1];
    fea3D.push_back(res);
    fea3DName.push_back("meanCurvature");
    std::cout<<"fea meanCurvature "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
//    qDebug()<<"fea3D size "<<fea3D.size()<<endl;
//    qDebug()<<"fea meanCurvature "<<fea3D[8]<<endl;

}

void Fea::setGaussianCurvature(GaussCurvature<MyMesh> &mesh,
                               std::vector<bool> &isVertexVisible)
{
    double res = 0.0;
    res = 0.0;
//    GaussCurvature<MyMesh> a(mesh);
    res = mesh.getGaussianCurvature(isVertexVisible);
    if(fea3D[1])
        res /= fea3D[1];
    fea3D.push_back(res);
    fea3DName.push_back("gaussianCurvature");
    std::cout<<"fea gaussianCurvature "<<res<<" fea3D size "<<fea3D.size()<<std::endl;
}

void Fea::setGaussianCurvature(std::vector< GaussCurvature<MyMesh>* > &a,
                               std::vector<bool> &isVertexVisible,
                               std::vector< std::vector<int> > &indiceArray)
{
    double res = 0.0;
//    freopen("D:/viewpoint/kmx/kxm.txt","w",stdout);
    for(int i=0;i<a.size();i++)
    {
        std::vector<bool> isVerVis;

        std::set<int> verIndice;
        for(int j=0;j<indiceArray[i].size();j++)
            verIndice.insert(indiceArray[i][j]);
        std::set<int>::iterator it = verIndice.begin();
        for(;it!=verIndice.end();it++)
            isVerVis.push_back(isVertexVisible[*it]);
//        std::cout<<"gauss ... hi "<<std::endl;
        res += a[i]->getGaussianCurvature(isVerVis);
//        std::cout<<res<<std::endl;
    }
    if(fea3D[1])
    res /= fea3D[1];
//    fclose(stdout);
    fea3D.push_back(res);
    fea3DName.push_back("gaussianCurvature");
    std::cout<<"fea gaussianCurvature "<<res<<" fea3D size "<<fea3D.size()<<std::endl;

}

void Fea::setMeshSaliency(std::vector< MeanCurvature<MyMesh> > &a, std::vector<GLfloat> &vertex, std::vector<bool> &isVertexVisible)
{
    double res = 0.0;
    double length = getDiagonalLength(vertex);
    std::vector<double> meanCurvature;
    double *nearDis = new double[vertex.size()/3];
//    MeanCurvature<MyMesh> a(mesh);
    double sigma[5] = {0.003*2.0,0.003*3.0,0.003*4.0,0.003*5.0,0.003*6.0};
    std::vector<double> meshSaliencyMiddle[5];
    double localMax[5];
    double gaussWeightedVal1,gaussWeightedVal2;
//    a.setMeanCurvature(meanCurvature);



    for(int j=0;j<5;j++)
    {
        localMax[j] = 0.0;
        for(int i=0;i<vertex.size();i+=3)
        {
            setNearDisMeshSaliency(vertex,i,length,sigma[j],nearDis);
            gaussWeightedVal1 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]);
            gaussWeightedVal2 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]*2.0);
            meshSaliencyMiddle[j].push_back(floatAbs(gaussWeightedVal1 - gaussWeightedVal2));
        }
        double max = meshSaliencyMiddle[j][0];
        double min = meshSaliencyMiddle[j][0];
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
        {
//            global max
            max = max > meshSaliencyMiddle[j][i] ? max : meshSaliencyMiddle[j][i];
//            used for normalize
            min = min > meshSaliencyMiddle[j][i] ? meshSaliencyMiddle[j][i] : min;
//            local max
            setNearDisMeshSaliency(vertex,i*3,length,sigma[j],nearDis);
            localMax[j] += getMeshSaliencyLocalMax(nearDis,vertex.size()/3,meshSaliencyMiddle[j]);
        }
        localMax[j] /= meshSaliencyMiddle[j].size();
//        normalize and set Si
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
            meshSaliencyMiddle[j][i] = (meshSaliencyMiddle[j][i] - min)/(max - min) *
                    (max - localMax[j])*(max - localMax[j]);
    }
//    set sum Si
    for(int i=0;i<meshSaliencyMiddle[0].size();i++)
        for(int j=1;j<5;j++)
            meshSaliencyMiddle[0][i] += meshSaliencyMiddle[j][i];

    for(int i=0;i<isVertexVisible.size();i++)
        if(isVertexVisible[i])
            res += meshSaliencyMiddle[0][i];
//    std::cout<<"fea meshSaliency ";
    fea3D.push_back(res);
    fea3DName.push_back("meshSaliency");
    std::cout <<" fea3D meshSaliency size "<<fea3D.size() << std::endl;

    delete []nearDis;
}

void Fea::setMeshSaliencyCompute(std::vector<MeanCurvature<MyMesh> > &a,
                                 std::vector<GLfloat> &vertex,
                                 std::vector<bool> &isVertexVisible,
                                 std::vector<std::vector<int> > &indiceArray)
{
    double res = 0.0;
    double length = getDiagonalLength(vertex);
    double *meanCurvature = new double[vertex.size()/3];
    memset(meanCurvature,0,sizeof(double)*vertex.size()/3);
    double *nearDis = new double[vertex.size()/3];
//    MeanCurvature<MyMesh> a(mesh);
    double sigma[5] = {0.003*2.0,0.003*3.0,0.003*4.0,0.003*5.0,0.003*6.0};
    std::vector<double> meshSaliencyMiddle[5];
    double localMax[5];
    double gaussWeightedVal1,gaussWeightedVal2;
//    a.setMeanCurvature(meanCurvature);

    for(int i=0;i<a.size();i++)
    {
        std::set<int> verIndice;
        std::vector<int> verVec;
        for(int j=0;j<indiceArray[i].size();j++)
            verIndice.insert(indiceArray[i][j]);
        std::set<int>::iterator it = verIndice.begin();
        for(;it!=verIndice.end();it++)
            verVec.push_back(*it);

//        MeanCurvature<MyMesh> a(vecMesh[i]);
        a[i].setMeanCurvature(meanCurvature,verVec);

    }

    for(int j=0;j<5;j++)
    {
        localMax[j] = 0.0;
        for(int i=0;i<vertex.size();i+=3)
        {
            setNearDisMeshSaliency(vertex,i,length,sigma[j],nearDis);
            gaussWeightedVal1 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]);
            gaussWeightedVal2 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]*2.0);
            meshSaliencyMiddle[j].push_back(floatAbs(gaussWeightedVal1 - gaussWeightedVal2));
        }
        double max = meshSaliencyMiddle[j][0];
        double min = meshSaliencyMiddle[j][0];
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
        {
//            global max
            max = max > meshSaliencyMiddle[j][i] ? max : meshSaliencyMiddle[j][i];
//            used for normalize
            min = min > meshSaliencyMiddle[j][i] ? meshSaliencyMiddle[j][i] : min;
//            local max
            setNearDisMeshSaliency(vertex,i*3,length,sigma[j],nearDis);
            localMax[j] += getMeshSaliencyLocalMax(nearDis,vertex.size()/3,meshSaliencyMiddle[j]);
        }
        localMax[j] /= meshSaliencyMiddle[j].size();
//        normalize and set Si
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
            meshSaliencyMiddle[j][i] = (meshSaliencyMiddle[j][i] - min)/(max - min) *
                    (max - localMax[j])*(max - localMax[j]);
        qDebug()<<"mesh saliency ... "<<j<<endl;
    }
//    set sum Si-
    for(int i=0;i<meshSaliencyMiddle[0].size();i++)
        for(int j=1;j<5;j++)
            meshSaliencyMiddle[0][i] += meshSaliencyMiddle[j][i];

    int num = isVertexVisible.size();
    for(int i=0;i<num;i++)
    {
        if(isVertexVisible[i])
            res += meshSaliencyMiddle[0][i];
    }
    fea3D.push_back(res);
    fea3DName.push_back("meshSaliency");
//    std::cout<<"fea meshSaliency ";
    //printf("fea meshSaliency %e\n",res);

    printf("fea meshSaliency %e",res);
    std::cout <<" fea3D size "<<fea3D.size() << std::endl;
    qDebug()<<"mesh saliency ... done"<<endl;
    delete []nearDis;
}

void Fea::setMeshSaliency(int t_case,// for debug can be used to output the mesh
                          std::vector<GLfloat> &vertex,
                          std::vector<bool> &isVertexVisible,
                          std::vector< MeanCurvature<MyMesh> > &a,
                          std::vector< std::vector<int> > &indiceArray)
{

    double res = 0.0;
    double length = getDiagonalLength(vertex);
//    std::vector<double> meanCurvature;
    double *meanCurvature = new double[vertex.size()/3];
    memset(meanCurvature,0,sizeof(double)*vertex.size()/3);
    double *nearDis = new double[vertex.size()/3];

    double sigma[5] = {0.003*2.0,0.003*3.0,0.003*4.0,0.003*5.0,0.003*6.0};
    std::vector<double> meshSaliencyMiddle[5];
    double localMax[5];
    double gaussWeightedVal1,gaussWeightedVal2;

    for(int i=0;i<a.size();i++)
    {
        std::set<int> verIndice;
        std::vector<int> verVec;
        for(int j=0;j<indiceArray[i].size();j++)
            verIndice.insert(indiceArray[i][j]);
        std::set<int>::iterator it = verIndice.begin();
        for(;it!=verIndice.end();it++)
            verVec.push_back(*it);

        a[i].setMeanCurvature(meanCurvature,verVec);

    }

    printf("set Mesh Saliency .... exImporter done\n");

    for(int j=0;j<5;j++)
    {
        localMax[j] = 0.0;
        for(int i=0;i<vertex.size();i+=3)
        {
            setNearDisMeshSaliency(vertex,i,length,sigma[j],nearDis);
            gaussWeightedVal1 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]);
            gaussWeightedVal2 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]*2.0);
            meshSaliencyMiddle[j].push_back(floatAbs(gaussWeightedVal1 - gaussWeightedVal2));
        }
        double max = meshSaliencyMiddle[j][0];
        double min = meshSaliencyMiddle[j][0];
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
        {
//            global max
            max = max > meshSaliencyMiddle[j][i] ? max : meshSaliencyMiddle[j][i];
//            used for normalize
            min = min > meshSaliencyMiddle[j][i] ? meshSaliencyMiddle[j][i] : min;
//            local max
            setNearDisMeshSaliency(vertex,i*3,length,sigma[j],nearDis);
            localMax[j] += getMeshSaliencyLocalMax(nearDis,vertex.size()/3,meshSaliencyMiddle[j]);
        }
        localMax[j] /= meshSaliencyMiddle[j].size();
//        normalize and set Si
        for(int i=0;i<meshSaliencyMiddle[j].size();i++)
            meshSaliencyMiddle[j][i] = (meshSaliencyMiddle[j][i] - min)/(max - min) *
                    (max - localMax[j])*(max - localMax[j]);

        printf("set MeshSaliency .... %d\n",j);
    }

//    set sum Si
    for(int i=0;i<meshSaliencyMiddle[0].size();i++)
        for(int j=1;j<5;j++)
            meshSaliencyMiddle[0][i] += meshSaliencyMiddle[j][i];

    for(int i=0;i<isVertexVisible.size();i++)
        if(isVertexVisible[i])
            res += meshSaliencyMiddle[0][i];
    fea3D.push_back(res);
    fea3DName.push_back("meshSaliency");
    printf("fea meshSaliency %e",res);
    std::cout <<" fea3D size "<<fea3D.size() << std::endl;
    delete []nearDis;
    delete []meanCurvature;
}

double Fea::setAbovePreference(double theta)
{
    double res = 0.0;
    double pi = asin(1.0)*2.0;
    res = exp(-(theta - pi/8.0*3.0)*(theta - pi/8.0*3.0)
                          / pi/4.0*pi/4.0);
    return res;
}

void Fea::setAbovePreference(glm::mat4 &model2,
                             glm::mat4 &model,
                             glm::mat4 &view)
{

        glm::vec4 z = glm::vec4(0.0,0.0,1.0,0.0);

        glm::vec4 x = glm::vec4(1.0,0.0,0.0,0.0);

        glm::vec4 y = glm::vec4(0.0,1.0,0.0,0.0);

        glm::vec4 mz = model*model2*z;
        glm::vec4 mx = model*model2*x;
        glm::vec4 my = model*model2*y;

    //    the theta between yyy and (0,1,0,1)
//        qDebug()<<"........."<<endl;
//        qDebug()<<yyy.x<<" "<<yyy.y<<" "<<yyy.z<<" "<<yyy.w<<endl;
        // ref http://stackoverflow.com/questions/21830340/understanding-glmlookat
        // I need center to eye // center - eye
        glm::vec4 lookAxis = glm::vec4(-view[0][2],-view[1][2],-view[2][2],0.f);

        float norm_mz = glm::dot(mz,mz);
        float norm_my = glm::dot(my,my);
        float norm_mx = glm::dot(mx,mx);

        float norm_lookAxis = glm::dot(lookAxis,lookAxis);

        float dotz = glm::dot(mz,lookAxis);
        float doty = glm::dot(my,lookAxis);
        float dotx = glm::dot(mx,lookAxis);

        double cosThetaz = dotz / sqrt(norm_mz) / sqrt(norm_lookAxis);
        double cosThetay = doty / sqrt(norm_my) / sqrt(norm_lookAxis);
        // added for absolute value
        cosThetay = floatAbs(cosThetay);
        double cosThetax = dotx / sqrt(norm_mx) / sqrt(norm_lookAxis);
        // added for absolute value
        cosThetax = floatAbs(cosThetax);

//        qDebug() << cosThetax << " "
//                 << cosThetay << " "
//                 << cosThetaz << endl;

        double thetaz = acos(cosThetaz);
//        double thetay = acos(cosThetay);
//        double thetax = acos(cosThetax);

        double resz = setAbovePreference(thetaz);
//        double resx = setAbovePreference(thetax);
//        double resy = setAbovePreference(thetay);

//        fea3D.push_back(resx);
//        fea3DName.push_back("abovePreference");
//        fea3D.push_back(resy);
//        fea3DName.push_back("abovePreference");
        fea3D.push_back(resz);
        fea3DName.push_back("abovePreference");

        std::cout<<"abovePreference "<<resz<<" fea3D size "<<fea3D.size()<<std::endl;
}

void Fea::setAbovePreference(glm::mat4 &modelZ, glm::mat4 &modelView)
{
    // still has a problem .... wait for finish
    glm::vec4 z = glm::vec4(0.0,0.0,1.0,0.0);
    glm::vec4 yyy = modelView*modelZ*z;


}

void Fea::setOutlierCount()
{
    // 看看渲染之后，有多少点是不在可视窗口内的
    double res = (double)render->p_outsidePointsNum / render->p_vertices.size();
    fea3D.push_back(res);
    fea3DName.push_back("outlierCount");
    std::cout << "outlier count "<< res <<" fea3D size "<<fea3D.size();
}
///
/// \brief Fea::setBoundingBox3D
///
/// bounding box中每一个渲染之后的边都会和现在的坐标轴有三个夹角
///
void Fea::setBoundingBox3D()
{
    double dotval = 0.0;
    double cosTheta = 0.0;
    double theta = 0.0;
    glm::vec4 axisx = glm::vec4(1,0,0,0);
    glm::vec4 axisy = glm::vec4(0,1,0,0);
    glm::vec4 axisz = glm::vec4(0,0,1,0);

    std::cout << "bounding box 3d "<<std::endl;
    // p_model_x x
    dotval = glm::dot(render->p_model_x,axisx);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisx));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "x x" << std::endl;
    // p_model_x y
    dotval = glm::dot(render->p_model_x,axisy);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisy));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
    //    std::cout << "x y" << std::endl;
    // p_model_x z
    dotval = glm::dot(render->p_model_x,axisz);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisz));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "x z" << std::endl;
    // p_model_y x
    dotval = glm::dot(render->p_model_y,axisx);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisx));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y x" << std::endl;
    // p_model_y y
    dotval = glm::dot(render->p_model_y,axisy);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisy));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y y" << std::endl;
    // p_model_y z
    dotval = glm::dot(render->p_model_y,axisz);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisz));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y z" << std::endl;
    // p_model_z x
    dotval = glm::dot(render->p_model_z,axisx);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisx));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z x" << std::endl;
    // p_model_z y
    dotval = glm::dot(render->p_model_z,axisy);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisy));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z y" << std::endl;
    // p_model_z z
    dotval = glm::dot(render->p_model_z,axisz);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisz));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z z" << std::endl;
    std::cout <<"bounding box done "<<" fea3D size "<<fea3D.size()<<std::endl;
}

void Fea::setBoundingBox3DAbs()
{
    double dotval = 0.0;
    double cosTheta = 0.0;
    double theta = 0.0;
    glm::vec4 axisx = glm::vec4(1,0,0,0);
    glm::vec4 axisy = glm::vec4(0,1,0,0);
    glm::vec4 axisz = glm::vec4(0,0,1,0);

    std::cout << "bounding box 3d "<<std::endl;
    // p_model_x x
    dotval = glm::dot(render->p_model_x,axisx);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisx));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "x x" << std::endl;
    // p_model_x y
    dotval = glm::dot(render->p_model_x,axisy);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisy));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
    //    std::cout << "x y" << std::endl;
    // p_model_x z
    dotval = glm::dot(render->p_model_x,axisz);
    cosTheta = dotval / (glm::length(render->p_model_x) * glm::length(axisz));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "x z" << std::endl;
    // p_model_y x
    dotval = glm::dot(render->p_model_y,axisx);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisx));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y x" << std::endl;
    // p_model_y y
    dotval = glm::dot(render->p_model_y,axisy);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisy));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y y" << std::endl;
    // p_model_y z
    dotval = glm::dot(render->p_model_y,axisz);
    cosTheta = dotval / (glm::length(render->p_model_y) * glm::length(axisz));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "y z" << std::endl;
    // p_model_z x
    dotval = glm::dot(render->p_model_z,axisx);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisx));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z x" << std::endl;
    // p_model_z y
    dotval = glm::dot(render->p_model_z,axisy);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisy));
    cosTheta = floatAbs(cosTheta);
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z y" << std::endl;
    // p_model_z z
    dotval = glm::dot(render->p_model_z,axisz);
    cosTheta = dotval / (glm::length(render->p_model_z) * glm::length(axisz));
    theta = acos(cosTheta);
    fea3D.push_back(theta);
    fea3DName.push_back("boundingBox");
//    std::cout << "z z" << std::endl;
    std::cout <<"bounding box done "<<" fea3D size "<<fea3D.size()<<std::endl;
}

void Fea::getColorDistribution()
{
#ifdef FOREGROUND
    double *hist = new double[NUM_Distribution];
    memset(hist,0,sizeof(double)*NUM_Distribution);

    int index = 0;

    for(int i=0;i<image2D.rows;i++)
        for(int j=0;j<image2D.cols;j++)
            if(mask.at<uchar>(i,j) != 255)
            {
                index = image2D.at<cv::Vec3b>(i,j)[0]>>5;
                index = (image2D.at<cv::Vec3b>(i,j)[1]>>5) + (index * 8);
                index = (image2D.at<cv::Vec3b>(i,j)[2]>>5) + (index * 8);
                hist[index]++;
            }

    for(int i=0;i<NUM_Distribution;i++)
    {
        fea2D.push_back(hist[i]);
        fea2DName.push_back("colorDistribution");
    }
    delete []hist;

#else
    double *hist = new double[NUM_Distribution];

    memset(hist,0,sizeof(double)*NUM_Distribution);
    int index = 0;

    for(int i=0;i<image2D.rows;i++)
        for(int j=0;j<image2D.cols;j++)
        {
            index = image2D.at<cv::Vec3b>(i,j)[0]>>5;
            index = (image2D.at<cv::Vec3b>(i,j)[1]>>5) + (index * 8);
            index = (image2D.at<cv::Vec3b>(i,j)[2]>>5) + (index * 8);
            hist[index]++;
        }

    for(int i=0;i<NUM_Distribution;i++)
    {
        fea2D.push_back(hist[i]);
        fea2DName.push_back("colorDistribution");
    }

    delete []hist;
#endif
//    qDebug()<<"color distribution done"<<endl;
    std::cout << "color distrbution done"<<" fea2D size "<<fea2D.size()<< std::endl;
}

void Fea::getHueCount()
{
    cv::Mat tmp = image2D32f3c;

//    qDebug()<<"image2D32f3c  "<<image2D32f3c.rows<<" "<<image2D32f3c.cols<<" "<<image2D32f3c.channels()<<endl;
//    qDebug()<<"image2D "<<image2D.rows<<" "<<image2D.cols<<endl;
    int histSize[1] = {20};
    int channels[1] = {0};
    float hranges[] = {0.f,360.f};
    const float *ranges[] = {hranges};
    float alpha = 0.05;


    cv::Mat mask0 = cv::Mat(image2D.rows,image2D.cols,CV_8UC1,cv::Scalar(0));
//    // it is useful to convert CV_8UC3 to CV_32FC3
//    image2D.convertTo(tmp,CV_32FC3,1/255.0);

//    cv::cvtColor(tmp,tmp,CV_BGR2HSV);

//      IplImage *saveImageDebug = new IplImage(mask);
//      cvSaveImage("/home/h005/Documents/vpDataSet/njuSample/model/640mask.jpg",saveImageDebug);

//      saveImageDebug = new IplImage(tmp);
//      cvSaveImage("/home/h005/Documents/vpDataSet/njuSample/model/640color.jpg",saveImageDebug);

    double valueRange[2] = {0.15,0.95};

    for(int i=0;i<tmp.rows;i++)
    {
        for(int j=0;j<tmp.cols;j++)
        {
#ifdef FOREGROUND
            if(mask.at<uchar>(i,j)!=255)
                if(tmp.at<cv::Vec3f>(i,j)[2] > valueRange[0] && tmp.at<cv::Vec3f>(i,j)[2] < valueRange[1] && tmp.at<cv::Vec3f>(i,j)[1]>0.2)
                {
                    mask0.at<uchar>(i,j) = 255;
                }
#else
            if(tmp.at<cv::Vec3f>(i,j)[2] > valueRange[0] && tmp.at<cv::Vec3f>(i,j)[2] < valueRange[1] && tmp.at<cv::Vec3f>(i,j)[1]>0.2)
            {
                mask0.at<uchar>(i,j) = 255;
            }
#endif
        }
    }
    cv::MatND hist;

    cv::calcHist(&tmp,1,channels,mask0,hist,
                 1,histSize,ranges);

    float max = 0.0;
    for(int i=0;i<histSize[0];i++)
        max = max < hist.at<float>(i) ? hist.at<float>(i) : max;

    float level = alpha * max;
    int count = 0;
    for(int i=0;i<histSize[0];i++)
        if(hist.at<float>(i) > level)
            count ++;
    double hueVal = histSize[0] - count;
    // 0
    fea2D.push_back(hueVal);
    fea2DName.push_back("HueCount");

//    tmp.release();
    mask0.release();
    hist.release();

//    qDebug()<<"hue count ... "<<hueVal<<" ... done"<<endl;
    std::cout << "hue count done "<<" fea2D size "<<fea2D.size()<< std::endl;

}

void Fea::getBlur()
{
#ifndef FOREGROUND
    cv::Mat tmp = gray;
//    qDebug()<<"blur "<<tmp.rows<<" "<<tmp.cols<<" "<<tmp.channels()<<endl;
//    image2D.copyTo(tmp);
//    cv::cvtColor(image2D,tmp,CV_BGR2GRAY);
    /*
     * Mat padded;                            //expand input image to optimal size
        int m = getOptimalDFTSize( I.rows );
        int n = getOptimalDFTSize( I.cols ); // on the border add zero pixels
        copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
        Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
        Mat complexI;
        merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
        dft(complexI, complexI);            // this way the result may fit in the source matrix
        split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
        Mat magI = planes[0];
     */
    float theta = 5.f;
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(tmp.rows);
    int n = cv::getOptimalDFTSize(tmp.cols);
    cv::copyMakeBorder(tmp,padded,0, m - tmp.rows,0,n - tmp.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),CV_32F)};
    cv::Mat complexI;
    cv::merge(planes,2,complexI);
    cv::dft(complexI,complexI);
    cv::split(complexI,planes);
    cv::magnitude(planes[0],planes[1],planes[0]);
    cv::Mat magl = planes[0];
    double blur = 0;
    for(int i=0;i<magl.rows;i++)
        for(int j=0;j<magl.cols;j++)
            if(magl.at<float>(i,j) > theta)
                blur ++;

    blur = blur / image2D.rows / image2D.cols;

    fea2D.push_back(blur);
    fea2DName.push_back("blur");
//    tmp.release();
    padded.release();
    complexI.release();
    planes[0].release();
    planes[1].release();
    magl.release();

#else
    return;
#endif

//    qDebug()<<" get blur ... "<<blur<<" ... done"<<endl;

}

void Fea::getContrast()
{
    double widthRatio = 0.98;

    double *hist = new double[256];
    memset(hist,0,sizeof(double)*256);

    int num = 0;

    for(int i=0;i<image2D.rows;i++)
        for(int j=0;j<image2D.cols;j++)
            for(int k = 0;k<3;k++)
#ifdef FOREGROUND
                if(mask.at<uchar>(i,j)!=255)
                {
                    hist[image2D.at<cv::Vec3b>(i,j)[k]]++;
                    num++;
                }
#else
                hist[image2D.at<cv::Vec3b>(i,j)[k]]++;
#endif

#ifndef FOREGROUND
    num = image2D.cols*image2D.rows*3;
#endif

    // num 的定义有错误，如果只考虑前景，则应该是前景像素的个数
//    int num = image2D.cols*image2D.rows*3;
    num = num*(1.0 - widthRatio);
    int count = 0;
    int from  = 0;
    int to = 255;
    while(count < num)
    {
        if(hist[from] < hist[to])
            count  = count + hist[from++];
        else
            count = count + hist[to--];
    }

    double contrast = to - from;

    // 1
    fea2D.push_back(contrast);
    fea2DName.push_back("contrast");
    delete []hist;
    std::cout << "contrast done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    qDebug()<<"get contrast ..."<<contrast<<" done"<<endl;

}

void Fea::getBrightness()
{
    cv::Mat tmp = gray;
//    image2D.copyTo(tmp);
//    cv::cvtColor(tmp,tmp,CV_BGR2GRAY);

    double count = 0.0;
    double sum = 0;
    for(int i=0;i<tmp.rows;i++)
        for(int j=0;j<tmp.cols;j++)
#ifdef FOREGROUND
            if(mask.at<uchar>(i,j)!=255)
            {
                sum += tmp.at<uchar>(i,j);
                count ++;
            }
    double brightness = sum / count;
#else
            sum += tmp.at<uchar>(i,j);
    double brightness = sum / tmp.rows / tmp.cols;
#endif

    // 2
    fea2D.push_back(brightness);
    fea2DName.push_back("brightness");
    std::cout << "brightness done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    tmp.release();

//    qDebug()<<"get Brightness ... "<<brightness<<endl;
}

void Fea::getRuleOfThird()
{
    double centroidRow  = 0;
    double centroidCol = 0;
    // ref http://mathworld.wolfram.com/GeometricCentroid.html
    double mess = 0;

//    qDebug()<<"getRuleOfThird .. "<<mask.rows<<" "<<mask.cols<<endl;

    for(int i=0;i<mask.rows;i++)
        for(int j=0;j<mask.cols;j++)
            if(mask.at<uchar>(i,j) != 255)
            {
                mess ++;
                centroidRow += i;
                centroidCol += j;
            }

    if(mess)
    {
        centroidRow /= mess;
        centroidCol /= mess;
    }

    //    scale to [0,1]
    if(mask.rows && mask.cols)
    {
        centroidRow /= mask.rows;
        centroidCol /= mask.cols;
    }

    double ruleOfThirdRow[2] = {1.0/3.0,2.0/3.0};
    double ruleOfThirdCol[2] = {1.0/3.0,2.0/3.0};

    double res = 100000.0;
    for(int i=0;i<2;i++)
        for(int j=0;j<2;j++)
        {
            double tmp = sqrt((centroidRow - ruleOfThirdRow[i])*(centroidRow - ruleOfThirdRow[i])
                              +(centroidCol - ruleOfThirdCol[j])*(centroidCol - ruleOfThirdCol[j]));
            res = res < tmp ? res : tmp;
        }

    fea2D.push_back(res);    
    fea2DName.push_back("ruleOfThird");
    std::cout << "rule of third done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    qDebug()<<"rule of third ... "<<res<<" ... done"<<endl;

}

void Fea::getLightingFeature()
{
    // feature lighting
    double fl = 0;
    // brightness of subject
    double bs = 0;
    // brightness of background
    double bb = 0;

    cv::Mat tmp = gray;
//    image2D.copyTo(tmp);

//    cv::cvtColor(tmp,tmp,CV_BGR2GRAY);

    for(int i=0;i<mask.rows;i++)
        for(int j=0;j<mask.cols;j++)
            if(mask.at<uchar>(i,j) != 255)
                bs += tmp.at<uchar>(i,j);
            else
                bb += tmp.at<uchar>(i,j);

    fl = floatAbs(log(bs/bb));

    fea2D.push_back(fl);
    fea2DName.push_back("lighting");

    std::cout << "lighting feature done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    tmp.release();

//    qDebug()<<"lighting feature .. "<<fl<<" ...done"<<endl;
}

//compute the GLCM of horizonal veritical and dialog
//and get the 4 features
//details in http://blog.csdn.net/carson2005/article/details/38442533
//the glcmMatirx has 16 values, so the size is 16*16
void Fea::setGLCM()
{
#ifndef FOREGROUND
    double glcmMatrix[GLCM_CLASS][GLCM_CLASS];
    double *glcm = new double[12];

    cv::Mat gray0 = cv::Mat(gray.size(),CV_8UC1);
    gray.copyTo(gray0);
    cv::Mat grade;
    grade = grade16(gray0);
    memset(glcm,0,sizeof(double)*12);
    int width = grade.cols;
    int height = grade.rows;


    //        horizontal
            memset(glcmMatrix,0,sizeof(glcmMatrix));
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                {
                    if((j+GLCM_DIS)<width)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i,j+GLCM_DIS)]++;
                    if((j-GLCM_DIS)>0)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i,j-GLCM_DIS)]++;
                }
            setGLCMfeatures(glcm,0,glcmMatrix);

    //        vertical
            memset(glcmMatrix,0,sizeof(glcmMatrix));
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                {
                    if((i+GLCM_DIS)<height)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i+GLCM_DIS,j)]++;
                    if((i-GLCM_DIS)>0)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i-GLCM_DIS,j)]++;
                }
            setGLCMfeatures(glcm,1,glcmMatrix);

    //        diagonal
            memset(glcmMatrix,0,sizeof(glcmMatrix));
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                {
                    if((i+GLCM_DIS)<height && (j+GLCM_DIS)<width)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i+GLCM_DIS,j+GLCM_DIS)]++;
                    if((i-GLCM_DIS)>0 && (j-GLCM_DIS)>0)
                        glcmMatrix[grade.at<uchar>(i,j)][grade.at<uchar>(i-GLCM_DIS,j-GLCM_DIS)]++;
                }
            setGLCMfeatures(glcm,2,glcmMatrix);


    gray0.release();
    grade.release();

    for(int i=0;i<12;i++)
    {
        fea2D.push_back(glcm[i]);
        fea2DName.push_back("glcm");
    }

    delete []glcm;
#else
    return;
#endif
//    qDebug()<<"glcm done"<<endl;
}

void Fea::setSaliency()
{
#ifndef FOREGROUND
    double thresh = 200;
    double salientArea = 0.0;

    double max = 0.0, min = 1e30;
//    cv::Mat gray;
//    cv::cvtColor(image2D,gray,CV_BGR2GRAY);

    // set hist
    double *hist = new double[256];
    memset(hist,0,sizeof(double)*256);
    for(int i=0;i<gray.rows;i++)
        for(int j=0;j<gray.cols;j++)
            hist[(int)gray.at<uchar>(i,j)]++;

    double **salient;
    salient = new double*[gray.rows];
    for(int i=0;i<gray.rows;i++)
    {
        salient[i] = new double[gray.cols];
        memset(salient[i],0,sizeof(double)*(gray.cols));
    }
    for(int i=0;i<gray.rows;i++)
        for(int j=0;j<gray.cols;j++)
        {
            for(int k=0;k<256;k++)
                salient[i][j] += floatAbs(hist[k] - hist[(int)gray.at<uchar>(i,j)]);
            max = max >salient[i][j]?max:salient[i][j];
            min = min<salient[i][j]?min:salient[i][j];
        }

    if(min == max)
    {
        salientArea = 0.0;
        fea2D.push_back(salientArea);
        fea2DName.push_back("salientArea");
        delete []hist;
        for(int i=0;i<gray.rows;i++)
            delete[] salient[i];
        delete [] salient;
        gray.release();
        return;
    }

    double a = 255.0/(max-min);
    double b = -255.0/(max-min)*min;

    for(int i=0;i<gray.rows;i++)
        for(int j=0;j<gray.cols;j++)
        {
            salient[i][j] = a*salient[i][j]+b;
            if(salient[i][j]>thresh)
                salientArea++;
        }

    salientArea = salientArea / gray.rows / gray.cols;
    fea2D.push_back(salientArea);
    fea2DName.push_back("salientArea");
    delete []hist;

    for(int i=0;i<gray.rows;i++)
        delete [] salient[i];
    delete []salient;
//    gray.release();
#else
    return;
#endif

//    qDebug()<<"set saliency done"<<endl;
}

void Fea::setPCA()
{
    for(int i=0;i<5;i++)
    {
        fea2D.push_back(pcaResult.at<float>(i,t_case));
        fea2DName.push_back("PCA");
    }
}

void Fea::getHog()
{
    // ref http://blog.csdn.net/yangtrees/article/details/7463431
    // ref http://blog.csdn.net/raodotcong/article/details/6239431
    // ref http://gz-ricky.blogbus.com/logs/85326280.html
    // ref http://blog.sciencenet.cn/blog-702148-762019.html
    cv::Mat gray0;
//    cv::cvtColor(image2D,gray0,CV_BGR2GRAY);
    int NUMbins = 9;
    double *hist = new double[NUMbins];
    memset(hist,0,sizeof(double)*NUMbins);
#ifdef FOREGROUND
    roundingBox(gray0);
    cv::resize(gray0,gray0,cv::Size(16,16));
#else
    cv::resize(gray,gray0,cv::Size(16,16));
#endif
    // widnowsSize,blockSize,blockStride,cellSize,numBins
    cv::HOGDescriptor d(cv::Size(16,16),cv::Size(8,8),cv::Size(4,4),cv::Size(4,4),NUMbins);

    std::vector<float> descriptorsValues;
    std::vector<cv::Point> locations;

    d.compute(gray0,descriptorsValues,cv::Size(0,0),cv::Size(0,0),locations);

    for(int i=0;i<descriptorsValues.size();i++)
        hist[i % NUMbins] += descriptorsValues[i];
    // 3 4 5 6 7 8 9 10 11
    for(int i=0;i<NUMbins;i++)
    {
        fea2D.push_back(hist[i]);
        fea2DName.push_back("hogHist");
    }

    delete hist;
    gray0.release();

    std::cout << "hog done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    qDebug()<<"hog done"<<endl;
}

void Fea::computePCA()
{
    cv::Mat pcaOriginal = cv::Mat(CoWidth*CoHeight,NUM,CV_32FC1);
    for(int i=0;i<NUM;i++)
    {
        cv::Mat gray = cv::imread(fileName.at(i).toStdString(),0);
        cv::resize(gray,gray,cv::Size(CoWidth,CoHeight));
        cv::Mat grayTmp = gray.reshape(1,gray.cols*gray.rows);

        cv::Mat tmp0 = pcaOriginal.col(i);

        grayTmp.convertTo(tmp0,CV_32FC1,1.0/255);

        gray.release();
    }

    cv::PCA pca0(pcaOriginal,cv::Mat(),CV_PCA_DATA_AS_COL,5);

    pcaResult = pca0.project(pcaOriginal);

    pcaOriginal.release();

}
///
/// \brief Fea::get2DTheta
/// 该函数共计算5个分量
/// 前两个计算的是投影之后的坐标轴与现在坐标轴（1,0,0）和（0,1,0）
/// 之间的最小夹角
/// 后三个是三个坐标轴投影之后两两之间的夹角
///
void Fea::get2DTheta()
{
    // Gyy 2b
    glm::vec3 axis_x = glm::vec3(1,0,0);
    glm::vec3 axis_y = glm::vec3(0,1,0);
    // 可能会出现零向量的问题，目前先当结果为2*pi来处理

    // p_model_x,p_model_y,p_model_z  transaction by MV without P
    // so we need multiply P matrix
    glm::vec4 mvp_model_x = m_projectionList[t_case] * render->p_model_x;
    glm::vec4 mvp_model_y = m_projectionList[t_case] * render->p_model_y;
    glm::vec4 mvp_model_z = m_projectionList[t_case] * render->p_model_z;
    glm::vec3 x_3d = glm::vec3(mvp_model_x[0],mvp_model_x[1],0);
    glm::vec3 y_3d = glm::vec3(mvp_model_y[0],mvp_model_y[1],0);
    glm::vec3 z_3d = glm::vec3(mvp_model_z[0],mvp_model_z[1],0);
    double theta = 0.0;
    double cosTheta = 0.0;
    // 12 13
    // compute minium theta between axis and the perspective vector
    if((x_3d.x == 0 && x_3d.y == 0) ||
            (y_3d.x == 0 && y_3d.y == 0) ||
            (z_3d.x == 0 && z_3d.y == 0))
    {
        fea2D.push_back( 2 * PI );
        fea2DName.push_back("2DTheta");
        fea2D.push_back( 2 * PI );
        fea2DName.push_back("2DTheta");
    }
    else{

        // y_axis x_3d
        cosTheta = glm::length(glm::dot(axis_y,x_3d)) / glm::length(axis_y) / glm::length(x_3d);
        theta = cosTheta;
        // y_axis y_3d
        cosTheta = glm::length(glm::dot(axis_y,y_3d)) / glm::length(axis_y) / glm::length(y_3d);
        theta = theta < cosTheta ? theta : cosTheta;
        // y_axis z_3d
        cosTheta = glm::length(glm::dot(axis_y,z_3d)) / glm::length(axis_y) / glm::length(z_3d);
        theta = theta < cosTheta ? theta : cosTheta;
        theta = acos(theta);
        fea2D.push_back(theta);
        fea2DName.push_back("2DTheta");
        // x_axis x_3d
        cosTheta = glm::length(glm::dot(axis_x,x_3d)) / glm::length(axis_x) / glm::length(x_3d);
        theta = cosTheta;
        // x_axis y_3d
        cosTheta = glm::length(glm::dot(axis_x,y_3d)) / glm::length(axis_x) / glm::length(y_3d);
        theta = theta < cosTheta ? theta : cosTheta;
        // x_axis z_3d
        cosTheta = glm::length(glm::dot(axis_x,z_3d)) / glm::length(axis_x) / glm::length(z_3d);
        theta = theta < cosTheta ? theta : cosTheta;
        theta = acos(theta);
        fea2D.push_back(theta);
        fea2DName.push_back("2DTheta");
    }

    //    double cosTheta = 0.0;
    //    double theta = 0.0;

    // thetas between three axis

    // 14 15 16
    cosTheta = 0.0;
    theta = 0.0;
    // x_3d y_3d
    if(glm::length(x_3d) ==0 || glm::length(y_3d)==0)
        theta = 0;
    else
    {
        cosTheta = glm::length(glm::dot(x_3d,y_3d)) / glm::length(x_3d) / glm::length(y_3d);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    // x_3d z_3d
    if(glm::length(x_3d) == 0 || glm::length(z_3d) == 0)
        theta = 0;
    else
    {
        cosTheta = glm::length(glm::dot(x_3d,z_3d)) / glm::length(x_3d) / glm::length(z_3d);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    // y_3d z_3d
    if(glm::length(y_3d)==0 || glm::length(z_3d) ==0 )
        theta = 0;
    else
    {
        cosTheta = glm::length(glm::dot(y_3d,z_3d)) /  glm::length(y_3d) / glm::length(z_3d);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    std::cout << "2DTheta done "<<" fea2D size "<<fea2D.size()<< std::endl;
}
///
/// \brief Fea::get2DThetaAbs
/// 该函数共计算5个分量
/// 前两个计算的是投影之后的坐标轴与现在坐标轴（1,0,0）和（0,1,0）
/// 之间的最小夹角 (absoute value without the difference of positive direction)
/// 后三个是三个坐标轴投影之后亮亮之间的夹角(absoute value without the difference of positive direction)
/// 我想了一下，还是感觉这个特征应该是属于3D特征的
///
void Fea::get2DThetaAbs()
{
    // bocaGyy 2b
    glm::vec3 axis_x = glm::vec3(1,0,0);
    glm::vec3 axis_y = glm::vec3(0,1,0);
    // 可能会出现零向量的问题，目前先当结果为2*pi来处理

    // p_model_x,p_model_y,p_model_z  transaction by MV without P
    // so we need multiply P matrix
    glm::vec4 mvp_model_x = m_projectionList[t_case] * render->p_model_x;
    glm::vec4 mvp_model_y = m_projectionList[t_case] * render->p_model_y;
    glm::vec4 mvp_model_z = m_projectionList[t_case] * render->p_model_z;
    glm::vec3 x_3d = glm::vec3(mvp_model_x[0],mvp_model_x[1],0);
    glm::vec3 y_3d = glm::vec3(mvp_model_y[0],mvp_model_y[1],0);
    glm::vec3 z_3d = glm::vec3(mvp_model_z[0],mvp_model_z[1],0);
    double theta = 0.0;
    double cosTheta = 0.0;
    // 12 13
    // compute minium theta between axis and the perspective vector
    if((x_3d.x == 0 && x_3d.y == 0) ||
            (y_3d.x == 0 && y_3d.y == 0) ||
            (z_3d.x == 0 && z_3d.y == 0))
    {
        fea2D.push_back( 2 * PI );
        fea2DName.push_back("2DTheta");
        fea2D.push_back( 2 * PI );
        fea2DName.push_back("2DTheta");
    }
    else{

        // y_axis x_3d
        cosTheta = glm::dot(axis_y,x_3d) / glm::length(axis_y) / glm::length(x_3d);
        cosTheta = floatAbs(cosTheta);
        theta = cosTheta;
        // y_axis y_3d
        cosTheta = glm::dot(axis_y,y_3d) / glm::length(axis_y) / glm::length(y_3d);
        cosTheta = floatAbs(cosTheta);
        theta = theta < cosTheta ? theta : cosTheta;
        // y_axis z_3d
        cosTheta = glm::dot(axis_y,z_3d) / glm::length(axis_y) / glm::length(z_3d);
        cosTheta = floatAbs(cosTheta);
        theta = theta < cosTheta ? theta : cosTheta;
        theta = acos(theta);
        fea2D.push_back(theta);
        fea2DName.push_back("2DTheta");
        // x_axis x_3d
        cosTheta = glm::dot(axis_x,x_3d) / glm::length(axis_x) / glm::length(x_3d);
        cosTheta = floatAbs(cosTheta);
        theta = cosTheta;
        // x_axis y_3d
        cosTheta = glm::dot(axis_x,y_3d) / glm::length(axis_x) / glm::length(y_3d);
        cosTheta = floatAbs(cosTheta);
        theta = theta < cosTheta ? theta : cosTheta;
        // x_axis z_3d
        cosTheta = glm::dot(axis_x,z_3d) / glm::length(axis_x) / glm::length(z_3d);
        cosTheta = floatAbs(cosTheta);
        theta = theta < cosTheta ? theta : cosTheta;
        theta = acos(theta);
        fea2D.push_back(theta);
        fea2DName.push_back("2DTheta");
    }

    //    double cosTheta = 0.0;
    //    double theta = 0.0;

    // thetas between three axis

    // 14 15 16
    cosTheta = 0.0;
    theta = 0.0;
    // x_3d y_3d
    if(glm::length(x_3d) ==0 || glm::length(y_3d)==0)
        theta = 0;
    else
    {
        cosTheta = glm::dot(x_3d,y_3d) / glm::length(x_3d) / glm::length(y_3d);
        cosTheta = floatAbs(cosTheta);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    // x_3d z_3d
    if(glm::length(x_3d) == 0 || glm::length(z_3d) == 0)
        theta = 0;
    else
    {
        cosTheta = glm::dot(x_3d,z_3d) / glm::length(x_3d) / glm::length(z_3d);
        cosTheta = floatAbs(cosTheta);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    // y_3d z_3d
    if(glm::length(y_3d)==0 || glm::length(z_3d) ==0 )
        theta = 0;
    else
    {
        cosTheta = glm::dot(y_3d,z_3d) /  glm::length(y_3d) / glm::length(z_3d);
        cosTheta = floatAbs(cosTheta);
        if(cosTheta > 1.0)
            theta = 0;
        else
            theta = acos(cosTheta);
    }
    fea2D.push_back(theta);
    fea2DName.push_back("2DTheta");
    std::cout << "2DTheta done "<<" fea2D size "<<fea2D.size()<< std::endl;
}



void Fea::roundingBox(cv::Mat &boxImage)
{
    int up,bottom,left,right;
    int sum = 0;
//    qDebug()<<"round box mask "<<mask.rows<<" "<<mask.cols<<endl;
    int back = mask.cols*255;
    // up
    for(int i=0;i<mask.rows;i++)
    {
        sum  = 0;
        for(int j=0;j<mask.cols;j++)
            sum += mask.at<uchar>(i,j);
        if(sum != back)
        {
            up = i;
            break;
        }
    }
//    qDebug()<<"roundBox up"<<endl;
    // bottom
    for(int i=mask.rows-1 ; i>=0 ; i--)
    {
        sum = 0;
        for(int j=0;j<mask.cols;j++)
            sum += mask.at<uchar>(i,j);
        if(sum != back)
        {
            bottom = i + 1;
            break;
        }
    }
//    qDebug()<<"roundBox bottom"<<endl;
    // left
    back = mask.rows*255;
    for(int i=0;i<mask.cols;i++)
    {
        sum = 0;
        for(int j=0;j<mask.rows;j++)
            sum += mask.at<uchar>(j,i);
        if(sum != back)
        {
            left = i;
            break;
        }
    }
//    qDebug()<<"roundBox left"<<endl;
    // right
    for(int i=mask.cols-1;i>=0;i--)
    {
        sum = 0;
        for(int j=0;j<mask.rows;j++)
            sum += mask.at<uchar>(j,i);
        if(sum != back)
        {
            right = i + 1;
            break;
        }
    }
//    qDebug()<<"roundBox right"<<endl;
//    qDebug()<<"bound box up bottom "<<up<<" "<<bottom<<endl;
//    qDebug()<<"bound box left right "<<left<<" "<<right<<endl;

    // image [left right) [up bottom)
//    Mat srcROI = src(Rect(0,0,src.cols/2,src.rows/2));
    cv::Mat tmp = gray(cv::Rect(left,up,right - left - 1,bottom - up - 1));
//    cv::namedWindow("gray");
//    cv::imshow("gray",gray);
    tmp.copyTo(boxImage);
//    qDebug()<<"bound box size "<<tmp.rows<<" "<<tmp.cols<<endl;

//    cv::namedWindow("boundingbox");
//    cv::imshow("boundingbox",boxImage);


}

void Fea::getColorEntropyVariance()
{
    double *hist = new double[NUM_Distribution];
    memset(hist,0,sizeof(double)*NUM_Distribution);
    double count = 0.0;
    int index = 0;

    for(int i=0;i<image2D.rows;i++)
        for(int j=0;j<image2D.cols;j++)
#ifdef FOREGROUND
            if(mask.at<uchar>(i,j) != 255)
            {
                index = image2D.at<cv::Vec3b>(i,j)[0]>>5;
                index = (image2D.at<cv::Vec3b>(i,j)[1]>>5) + (index * 8);
                index = (image2D.at<cv::Vec3b>(i,j)[2]>>5) + (index * 8);
                hist[index]++;
                count ++;
            }
#else
        {
            index = image2D.at<cv::Vec3b>(i,j)[0]>>5;
            index = (image2D.at<cv::Vec3b>(i,j)[1]>>5) + (index * 8);
            index = (image2D.at<cv::Vec3b>(i,j)[2]>>5) + (index * 8);
            hist[index]++;
            count ++;
        }
#endif
    double mean = 0.0;
    // normalize
    for(int i=0;i<NUM_Distribution;i++)
    {
        hist[i] = hist[i] / count;
        mean += hist[i] * i;
    }
    mean /= NUM_Distribution;

    double entropy = 0.0;
    // entropy
    for(int i=0;i<NUM_Distribution;i++)
        if(hist[i])
            entropy += hist[i] * log2(hist[i]);
    entropy = -entropy;

//    // variance
//    double variance = 0.0;
//    for(int i=0;i<NUM_Distribution;i++)
//        variance += hist[i] * (i - mean) * (i - mean);

    // dirstribution as depth distribution
    double dis = 0.0;
    for(int i=0;i<NUM_Distribution;i++)
        dis += hist[i] * hist[i];
    dis = 1 - dis;

    // 17 18 19
    fea2D.push_back(entropy);
    fea2DName.push_back("EntropyVariance");
//    fea2D.push_back(variance);
//    fea2DName.push_back("EntropyVariance");
    fea2D.push_back(dis);
    fea2DName.push_back("EntropyVariance");
    std::cout << "color Entropy variance done "<<" fea2D size "<<fea2D.size()<< std::endl;
//    for(int i=0;i<NUM_Distribution;i++)
//        fea2D.push_back(hist[i]);
    delete []hist;
}
///
/// \brief Fea::getColorInfo
/// this function was created to compute another color info
/// such as RGB values mean
/// HSV vlaues C1 in HSV space
/// Hue, histogram (5 bins) and entropy
/// Saturation, histogram (3 bins) and entropy
///
void Fea::getColorInfo()
{
    // rgb mean
    double bgrMean[3] = {0,0,0};
    for(int i=0;i<image2D.rows;i++)
        for(int j=0;j<image2D.cols;j++)
        {
            bgrMean[0] += (double)image2D.at<cv::Vec3b>(i,j)[0];
            bgrMean[1] += (double)image2D.at<cv::Vec3b>(i,j)[1];
            bgrMean[2] += (double)image2D.at<cv::Vec3b>(i,j)[2];
        }
    double num = (double)(image2D.rows * image2D.cols);
    for(int i=0;i<2;i++)
        bgrMean[i] /= num;

    fea2D.push_back(bgrMean[0]);;
    fea2D.push_back(bgrMean[1]);
    fea2D.push_back(bgrMean[2]);

    fea2DName.push_back("color blue mean");
    fea2DName.push_back("color green mean");
    fea2DName.push_back("color red mean");

    // compute hsv featues
    cv::Mat imagehsv;
    cv::cvtColor(image2D,imagehsv,CV_BGR2HSV);

    //    compute Hue histogram
    int hbins = 5;
    int histSize[] = {hbins};

    float hranges[] = {0,180};

    const float* ranges[] = {hranges};
    cv::MatND hist;
    int channels[] = {0};

    cv::calcHist(&imagehsv,
                 1,
                 channels,
                 cv::Mat(),
                 hist,
                 1,
                 histSize,
                 ranges,
                 true,
                 false);

//    qDebug() << "hHist " << endl;
    for(int h=0;h<hbins;h++)
    {
//        qDebug() << hist.at<float>(h) << endl;
        hist.at<float>(h) /= num;
        fea2D.push_back(hist.at<float>(h));
        fea2DName.push_back("HueHist");
    }

    double entropy = 0.0;
    for(int i=0;i<hbins;i++)
        if(hist.at<float>(i))
            entropy += hist.at<float>(i) * log2(hist.at<float>(i));
    entropy = -entropy;
    fea2D.push_back(entropy);
    fea2DName.push_back("HueEntropy");

//    qDebug() << entropy << endl;

    int sbins = 3;
    int sHistSize[] = {sbins};

    float sranges[] = {0,256};

    const float *Sranges[] = {sranges};

    cv::MatND shist;
    channels[0] = 1;

    cv::calcHist(&imagehsv,
                 1,
                 channels,
                 cv::Mat(),
                 shist,
                 1,
                 sHistSize,
                 Sranges,
                 true,
                 false);

//    qDebug() << "Saturation" << endl;

    for(int h =0 ;h<sbins;h++)
    {
//        qDebug() << shist.at<float>(h) << endl;
        shist.at<float>(h) /= num;
        fea2D.push_back(shist.at<float>(h));
        fea2DName.push_back("SaturationHist");
    }

    entropy = 0.0;
    for(int i=0;i<sbins;i++)
        if(shist.at<float>(i))
            entropy += shist.at<float>(i) * log2(shist.at<float>(i));
    entropy = -entropy;
    fea2D.push_back(entropy);
    fea2DName.push_back("SaturationEntropy");
//    qDebug() << entropy << endl;
}

void Fea::getBallCoord()
{
    // ref https://zh.wikipedia.org/wiki/%E7%90%83%E5%BA%A7%E6%A8%99%E7%B3%BB
    glm::mat4 mv = m_viewList[t_case] *  m_modelList[t_case];
    glm::vec4 camera = glm::vec4(0,0,-1,0);
    camera = glm::inverse(mv) * camera;
    camera[3] = 0.0;
//    double r = glm::length(camera);
    double theta = PI + atan(sqrt(camera[0] * camera[0] + camera[1] * camera[1]) / camera[2]); 
    double fani;
    if(camera[0] == 0)
        fani = 0;
    else
    fani = atan(camera[1] / camera[0]);
//    std::cout << "ball coord" << std::endl;
//    std::cout << theta << " " << fani << std::endl;
    fea3D.push_back(theta);
    fea3DName.push_back("ballCoord");
    fani = floatAbs(fani);
    fea3D.push_back(fani);
    fea3DName.push_back("ballCoord");
    std::cout << "ballCoord done "<<" fea3D size "<<fea3D.size()<< std::endl;
}

///
/// \brief Fea::roundingBox2D  bounding box of 2D image after rendering
/// \param up, up boundary
/// \param bottom, bottom boundary
/// \param left, left boundary
/// \param right, right boundary
///
void Fea::roundingBox2D(int &up, int &bottom, int &left, int &right)
{
    up = bottom = left = right = 0;
    int sum = 0;
    int back = image.cols * 230;
    // up
    for(int i=0;i<image.rows;i++)
    {
        sum  = 0;
        for(int j=0;j<image.cols;j++)
            sum += image.at<uchar>(i,j);
        if(sum <= back)
        {
            up = i;
            break;
        }
    }
    // bottom
    for(int i=image.rows-1 ; i>=0 ; i--)
    {
        sum = 0;
        for(int j=0;j<image.cols;j++)
            sum += image.at<uchar>(i,j);
        if(sum <= back)
        {
            bottom = i + 1;
            break;
        }
    }
    // left
    back = image.rows*230;
    for(int i=0;i<image.cols;i++)
    {
        sum = 0;
        for(int j=0;j<image.rows;j++)
            sum += image.at<uchar>(j,i);
        if(sum <= back)
        {
            left = i;
            break;
        }
    }
    // right
    for(int i=image.cols-1;i>=0;i--)
    {
        sum = 0;
        for(int j=0;j<image.rows;j++)
            sum += image.at<uchar>(j,i);
        if(sum <= back)
        {
            right = i + 1;
            break;
        }
    }
}

double Fea::getMeshSaliencyLocalMax(double *nearDis, int len, std::vector<double> meshSaliency)
{
    //可能会有bug,nearDis[0]如果为0的话，赋值是没有意义的
//    double max = meshSaliency[0];
    // meshSaliency >= 0
    double max = 0;
    for(int i=0;i<len;i++)
        if(nearDis[i])
            max = max > meshSaliency[i] ? max : meshSaliency[i];
    return max;
}

double Fea::getGaussWeightedVal(double meanCur, double *nearDis, int len, double sigma)
{
    double numerator = 0.0,denominator = 0.0;
    double expVal = 0.0;
    sigma = 2.0*sigma*sigma;
    for(int i=0;i<len;i++)
    {
        expVal = exp(- nearDis[i] / sigma);
        numerator += meanCur*expVal;
        denominator += expVal;
    }
    return numerator/denominator;
}

double Fea::getDiagonalLength(std::vector<GLfloat> &vertex)
{
//    top t 0
//    bottom b 1  tb
//    left l 0
//    right r 1   lr
//    front f 0
//    behind b 1  fb
//    named as v+lf+tb+fb
//    if vxyz + vpqr = v111 then they are diagonal
    double v_000[3] = {vertex[0],vertex[1],vertex[2]};
    double v_001[3] = {vertex[0],vertex[1],vertex[2]};
    double v_010[3] = {vertex[0],vertex[1],vertex[2]};
    double v_011[3] = {vertex[0],vertex[1],vertex[2]};
    double v_100[3] = {vertex[0],vertex[1],vertex[2]};
    double v_101[3] = {vertex[0],vertex[1],vertex[2]};
    double v_110[3] = {vertex[0],vertex[1],vertex[2]};
    double v_111[3] = {vertex[0],vertex[1],vertex[2]};
    for(int i=3;i<vertex.size();i+=3)
    {
        vertexBoundBox(v_000,vertex,i,0);
        vertexBoundBox(v_001,vertex,i,1);
        vertexBoundBox(v_010,vertex,i,2);
        vertexBoundBox(v_011,vertex,i,3);
        vertexBoundBox(v_100,vertex,i,4);
        vertexBoundBox(v_101,vertex,i,5);
        vertexBoundBox(v_110,vertex,i,6);
        vertexBoundBox(v_111,vertex,i,7);
    }
    double diag[4];
    diag[0] = sqrt((v_000[0]-v_111[0])*(v_000[0]-v_111[0])
            +(v_000[1]-v_111[1])*(v_000[1]-v_111[1])
            +(v_000[2]-v_111[2])*(v_000[2]-v_111[2]));
    diag[1] = sqrt((v_001[0]-v_110[0])*(v_001[0]-v_110[0])
            +(v_001[1]-v_110[1])*(v_001[1]-v_110[1])
            +(v_001[2]-v_110[2])*(v_001[2]-v_110[2]));
    diag[2] = sqrt((v_010[0]-v_101[0])*(v_010[0]-v_101[0])
            +(v_010[1]-v_101[1])*(v_010[1]-v_101[1])
            +(v_010[2]-v_101[2])*(v_010[2]-v_101[2]));
    diag[3] = sqrt((v_011[0]-v_100[0])*(v_011[0]-v_100[0])
            +(v_011[1]-v_100[1])*(v_011[1]-v_100[1])
            +(v_011[2]-v_100[2])*(v_011[2]-v_100[2]));

    double max = 0.0;
    for(int i=0;i<4;i++)
        max = max<diag[i]?diag[i]:max;
    return max;
}

void Fea::setNearDisMeshSaliency(std::vector<GLfloat> &vertex, int index, double len, double sigma, double *nearDis)
{
    double dis = len*2.0*sigma;
    //avoid for sqrt
    dis = dis*dis;
    double disV0V1 = 0.0;
    glm::vec3 v0 = glm::vec3(vertex[index],vertex[index+1],vertex[index+2]);
    glm::vec3 v1;
    for(int i=0;i<vertex.size();i+=3)
    {
        v1 = glm::vec3(vertex[i],vertex[i+1],vertex[i+2]);
        v1 = v1-v0;
        v1 = v1*v1;
        disV0V1 = v1.x+v1.y+v1.z;
        if(disV0V1 > dis)
            nearDis[i/3] = 0.0;
        else
            nearDis[i/3] = disV0V1;
    }
}

void Fea::vertexBoundBox(double *v, std::vector<GLfloat> &vertex, int i, int label)
{
    switch (label) {
    case 0:
        if(vertex[i] >= v[0] &&
           vertex[i+1] >= v[1] &&
           vertex[i+2] >= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 1:
        if(vertex[i] >= v[0] &&
           vertex[i+1] >= v[1] &&
           vertex[i+2] <= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 2:
        if(vertex[i] >= v[0] &&
           vertex[i+1] <= v[1] &&
           vertex[i+2] >= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 3:
        if(vertex[i] >= v[0] &&
           vertex[i+1] <= v[1] &&
           vertex[i+2] <= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 4:
        if(vertex[i] <= v[0] &&
           vertex[i+1] >= v[1] &&
           vertex[i+2] >= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 5:
        if(vertex[i] <= v[0] &&
           vertex[i] >= v[1] &&
           vertex[i] <= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 6:
        if(vertex[i] <= v[0] &&
           vertex[i] <= v[1] &&
           vertex[i] >= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    case 7:
        if(vertex[i] <= v[0] &&
           vertex[i+1] <= v[1] &&
           vertex[i+2] <= v[2])
        {
            v[0] = vertex[i];
            v[1] = vertex[i+1];
            v[2] = vertex[i+2];
        }
        break;
    default:
        break;
    }
}

bool Fea::getCurvature(CvPoint2D64f *a, CvPoint2D64f *b, CvPoint2D64f *c, double &cur)
{
    double r = 0;
    cur = 0.0;
    if(getR(a,b,c,r))
    {
        cur = 1.0/r;
        return true;
    }
    else
        return false;
}

void Fea::readCvSeqTest(CvSeq *seq)
{
    qDebug()<<"readCvSeqTest"<<endl;
    for(int i=0;i<seq->total;i++)
    {
        CvPoint *point = CV_GET_SEQ_ELEM(CvPoint,
                                         seq,i);
        qDebug()<<point->x<<point->y<<endl;
    }
}

double Fea::getArea2D(CvPoint2D64f *a, CvPoint2D64f *b, CvPoint2D64f *c)
{
    CvPoint2D64f ab = cvPoint2D64f(b->x - a->x, b->y - a->y);
    CvPoint2D64f ac = cvPoint2D64f(c->x - a->x, c->y - a->y);
    double area = ab.x * ac.y - ab.y * ac.x;
    area = area > 0 ? area : -area;
    area /= 2.0;
    return area;
}

double Fea::getArea3D(CvPoint3D64f *a, CvPoint3D64f *b, CvPoint3D64f *c)
{
    CvPoint3D64f ab = cvPoint3D64f(b->x - a->x, b->y - a->y, b->z - a->z);
    CvPoint3D64f ac = cvPoint3D64f(c->x - a->x, c->y - a->y, c->z - a->z);
    double area = (ab.y*ac.z - ac.y*ab.z)*(ab.y*ac.z - ac.y*ab.z)
             + (ac.x*ab.z - ab.x*ac.z)*(ac.x*ab.z - ab.x*ac.z)
            + (ab.x*ac.y - ac.x*ab.y)*(ab.x*ac.y - ac.x*ab.y);
    area = sqrt(area);
    area /= 2.0;
    return area;
}


double Fea::getDis3D(std::vector<float> &vertex,
                   int i1, int i2)
{
    double dx = (vertex[i1*3]-vertex[i2*3]);
    double dy = (vertex[i1*3+1]-vertex[i2*3+1]);
    double dz = (vertex[i1*3+2]-vertex[i2*3+2]);
    return sqrt(dx*dx+dy*dy+dz*dz);
}

double Fea::getDis2D(CvPoint2D64f *a, CvPoint2D64f *b)
{
    double dx = (a->x-b->x);
    double dy = (a->y-b->y);
    return sqrt(dx*dx+dy*dy);
}

double Fea::cosVal3D(std::vector<float> &vertex,
                   int i0, int i1, int i2)
{
    double dotVal = (vertex[i1*3]-vertex[i2*3])*(vertex[i0*3]-vertex[i2*3])
            +(vertex[i1*3+1]-vertex[i2*3+1])*(vertex[i0*3+1]-vertex[i2*3+1])
            +(vertex[i1*3+2]-vertex[i2*3+2])*(vertex[i0*3+2]-vertex[i2*3+2]);
    double va = (vertex[i1*3]-vertex[i2*3])*(vertex[i1*3]-vertex[i2*3])
            +(vertex[i1*3+1]-vertex[i2*3+1])*(vertex[i1*3+1]-vertex[i2*3+1])
            +(vertex[i1*3+2]-vertex[i2*3+2])*(vertex[i1*3+2]-vertex[i2*3+2]);
    va = sqrt(va);
    double vb = (vertex[i0*3]-vertex[i2*3])*(vertex[i0*3]-vertex[i2*3])
            + (vertex[i0*3+1]-vertex[i2*3+1])*(vertex[i0*3+1]-vertex[i2*3+1])
            + (vertex[i0*3+2]-vertex[i2*3+2])*(vertex[i0*3+2]-vertex[i2*3+2]);
    vb = sqrt(vb);
    return dotVal/va/vb;
}

double Fea::cosVal2D(CvPoint2D64f *a, CvPoint2D64f *b, CvPoint2D64f *c)
{
    double dotVal = (a->x-c->x)*(b->x-c->x)
            +(a->y-c->y)*(b->y-c->y);
    double va = (a->x-c->x)*(a->x-c->x)
            +(a->y-c->y)*(a->y-c->y);
    double vb = (b->x-c->x)*(b->x-c->x)
            +(b->y-c->y)*(b->y-c->y);

    va = sqrt(va);
    vb = sqrt(vb);
//    std::cout<<"...cosVal2D "<<dotVal<<" "<<va<<" "<<vb<<std::endl;
    if(!va)
        return 0;
    if(!vb)
        return 0;
    return dotVal/va/vb;
}

bool Fea::getR(CvPoint2D64f *a, CvPoint2D64f *b, CvPoint2D64f *c, double &r)
{
//        area = a*b*sin(ACB)/2
//        area = a*b*sin(ACB)/2
//        the first vertex is A the edge is a(BC)
//        the second vertex is B the edge is b(AC)
//        the third vertex is C the edge is c(AB)
    double c0 = getDis2D(a,b);
    double cosACB = cosVal2D(a,b,c);
    double sinACB = sqrt(1.0 - cosACB*cosACB);
//    std::cout<<c0<<" "<<sinACB<<std::endl;
    if(!sinACB)
        return false;
    r = c0/2.0/sinACB;
    return true;
}

void Fea::normalizeHist(double *hist,double step,int num)
{
    double area = 0.0;
    for(int i=0;i<num;i++)
        area += hist[i]*step;
    for(int i=0;i<num;i++)
        hist[i] /= area;
}

void Fea::initial()
{
}

void Fea::setMvpPara(QString matrixFile)
{
    this->matrixPath = matrixFile;
    qDebug()<<matrixFile<<endl;
    freopen(matrixPath.toStdString().c_str(),"r",stdin);
    QString tmp = QDir::cleanPath(path);
    QFileInfo imgPathHelper(tmp);
    char tmpss[200];
    float tmpNum;
    QString imgPath = imgPathHelper.absolutePath();
    imgPath.append("/imgs/");

    while(scanf("%s",tmpss)!=EOF)
    {
//        QString tmpPath = path;
//        tmp = QDir::cleanPath(tmpPath);

//        int pos = tmp.lastIndexOf('/');
//        tmp = tmp.left(pos+1);
//        tmp = QDir::cleanPath(tmp.append(QString(tmpss)));
        QString ssHelper(tmpss);
        QFileInfo imgFile(ssHelper);
        imgFile.absoluteDir();
        QString filename = imgFile.fileName();
        tmp = imgPath;
        tmp.append(filename);

        fileName.push_back(tmp);

        glm::mat4 m,v,p;
        for(int i=0;i<16;i++)
        {
            scanf("%f",&tmpNum);
            m[i%4][i/4] = tmpNum;
        }
        this->m_modelList.push_back(m);
        this->m_viewList.push_back(v);

#ifdef NoProjection

        p = glm::perspective(glm::pi<float>() / 2, 1.f, 0.1f, 100.f);
        this->m_projectionList.push_back(p);

#else
        for(int i=0;i<16;i++)
        {
            scanf("%f",&tmpNum);
            p[i%4][i/4] = tmpNum;
        }
        this->m_projectionList.push_back(p);
#endif  
    }
    NUM = fileName.size();
}

void Fea::printOut(int mode)
{
    if(mode == 3)
    {
        // output 3D feature to .3df file
        freopen(output3D.toStdString().c_str(),"a+",stdout);
    #ifdef CHECK
        // output the order number
        printf("%s\n",QString::number(t_case).toStdString().c_str());
    #else
        // output the fileName
        printf("%s\n",fileName.at(t_case).toStdString().c_str());
    #endif

        // output 3d feature
        for(int i=0;i<fea3D.size();i++)
        {
            // 8 mean curvature
            // 9 gauss curvature
            // 10 mesh saliency did not count now
            if(i==8 || i== 9)
                printf("%e ",fea3D[i]);
            else
                printf("%lf ",fea3D[i]);
        }
        printf("\n");
        fclose(stdout);

    }
    if(mode == 2)
    {
        // output 2D feature to .2df file
        freopen(output2D.toStdString().c_str(),"a+",stdout);
    #ifdef CHECK
        // output the order number
        printf("%s\n",QString::number(t_case).toStdString().c_str());
    #else
        // output the fileName
        printf("%s\n",fileName.at(t_case).toStdString().c_str());
    #endif
        // output 2D feature
        for(int i=0;i<fea2D.size();i++)
            printf("%lf ",fea2D[i]);

        printf("\n");
        fclose(stdout);
    }
    if(mode  == 0)
    {
        // output 3D feature to .3df file
        freopen(output3D.toStdString().c_str(),"a+",stdout);
    #ifdef CHECK
        // output the order number
        printf("%s\n",QString::number(t_case).toStdString().c_str());
    #else
        // output the fileName
        printf("%s\n",fileName.at(t_case).toStdString().c_str());
    #endif

        // output 3d feature
        for(int i=0;i<fea3D.size();i++)
        {
            // 8 mean curvature
            // 9 gauss curvature
            // 10 mesh saliency did not count now
            if(i==8 || i== 9)
                printf("%e ",fea3D[i]);
            else
                printf("%lf ",fea3D[i]);
        }
        printf("\n");
        fclose(stdout);

        // output 2D feature to .2df file
        freopen(output2D.toStdString().c_str(),"a+",stdout);
    #ifdef CHECK
        // output the order number
        printf("%s\n",QString::number(t_case).toStdString().c_str());
    #else
        // output the fileName
        printf("%s\n",fileName.at(t_case).toStdString().c_str());
    #endif
        // output 2D feature
        for(int i=0;i<fea2D.size();i++)
            printf("%lf ",fea2D[i]);

        printf("\n");
        fclose(stdout);
    }

    // output log file
    freopen(mmPath.toStdString().c_str(),"w",stdout);
#ifdef CHECK
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
            printf("%f ",m_view[i][j]);
        printf("\n");
    }

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
            printf("%f ",m_abv[i][j]);
        printf("\n");
    }

    printf("%d %d\n",t_case+1,NUM);
#else

    // rearrange the model needless to printout the parameters

//    for(int i=0;i<4;i++)
//    {
//        for(int j=0;j<4;j++)
//            printf("%f ",m_abv[i][j]);
//        printf("\n");
//    }

    printf("%d\n",t_case+1);


#endif

    fclose(stdout);
    image.release();
}

void Fea::printFeaName(int mode)
{
    freopen(outputFeaName.toStdString().c_str(),"a+",stdout);

    for(int j=0;j<fea2DName.size();j++)
        std::cout << fea2DName[j] << std::endl;

    for(int i=0;i<fea3DName.size();i++)
        std::cout << fea3DName[i] << std::endl;

    fclose(stdout);

    freopen(output2dFeaName.toStdString().c_str(),"a+",stdout);
    for(int j=0;j<fea2DName.size();j++)
        std::cout << fea2DName[j] << std::endl;
    fclose(stdout);

    freopen(output3dFeaName.toStdString().c_str(),"a+",stdout);
    for(int j=0;j<fea3DName.size();j++)
        std::cout << fea3DName[j] << std::endl;
    fclose(stdout);

}

void Fea::computeModel(glm::mat4 &m_view_tmp,glm::mat4 &m_model_tmp)
{
    // rotate with x axis in openGL
//    float angle_x = 180.0/MAX_LEN;
    float angle_x = glm::pi<float>()/MAX_LEN;
    // rotate with z axis in model
//    float angle_z = 2.0*180.0/MAX_LEN;
    float angle_z = 2.0 * glm::pi<float>()/MAX_LEN;

    int tmp = t_case / MAX_LEN;
    float angle = - angle_x * tmp;
    glm::mat4 rotateX = glm::rotate(glm::mat4(1.f),angle,glm::vec3(1.0,0.0,0.0));
    m_view_tmp = m_view * rotateX;
    tmp = t_case % MAX_LEN;
    angle = - angle_z * tmp;
    glm::mat4 rotateZ = glm::rotate(glm::mat4(1.f),angle,glm::vec3(0.0,0.0,1.0));
    m_model_tmp = m_model * rotateZ;
//    return rotateX;

}

void Fea::computeModel(glm::mat4 &m_model_tmp)
{
    // rotate with x axis in openGL
//    float angle_x = 180.0/MAX_LEN;
    float angle_x = glm::pi<float>()/MAX_LEN;
    // rotate with z axis in model
//    float angle_y = 2.0*180.0/MAX_LEN;
    float angle_y = 2.0 * glm::pi<float>()/MAX_LEN;

    int tmp = t_case / MAX_LEN;
    float angle = angle_x * tmp;
    qDebug()<<"computeModel.... x "<<angle<<endl;
    glm::mat4 rotateX = glm::rotate(glm::mat4(1.f),angle,glm::vec3(1.0,0.0,0.0));
    tmp = t_case % MAX_LEN;
    angle = angle_y * tmp;
    qDebug()<<"computeModel.... y "<<angle<<endl;
    glm::mat4 rotateY = glm::rotate(glm::mat4(1.f),angle,glm::vec3(0.0,1.0,0.0));
    m_model_tmp = m_model * rotateX * rotateY;

}

/*
double *Fea::getFeaArray() const
{
    return feaArray;
}

void Fea::setFeaArray(double *value)
{
    feaArray = value;
}
*/
void Fea::showImage()
{
    render->showImage();
}

Fea::Fea()
{

}


typedef long double LD;
double Fea::getContourCurvature(const std::vector<cv::Point2d> &points, int target)
{
    assert(points.size() == 3);

    double T[3];
    for (int i = 0; i < 3; i++) {
        double t = cv::norm(points[target] - points[i]);
        T[i] = target < i ? t : -t;
    }
    cv::Mat M(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) {
        M.at<double>(i, 0) = 1;
        M.at<double>(i, 1) = T[i];
        M.at<double>(i, 2) = T[i] * T[i];
    }
    cv::Mat invM = M.inv();

    cv::Mat X(3, 1, CV_64F), Y(3, 1, CV_64F);
    for (int i = 0; i < 3; i++) {
        X.at<double>(i, 0) = points[i].x;
        Y.at<double>(i, 0) = points[i].y;
    }

    cv::Mat a, b;
    a = invM * X;
    b = invM * Y;

    LD up = (LD)2 * (a.at<double>(1, 0) * b.at <double>(2, 0) - a.at<double>(2, 0) * b.at <double>(1, 0));
    LD down = pow((LD)a.at<double>(1, 0) * a.at<double>(1, 0) + (LD)b.at <double>(1, 0) * b.at <double>(1, 0), 1.5);
    LD frac = up / down;

    return (double)frac;
}

cv::Mat Fea::grade16(cv::Mat gray)
{
    for(int i=0;i<gray.rows;i++)
        for(int j=0;j<gray.cols;j++)
            gray.at<uchar>(i,j) = (gray.at<uchar>(i,j) & 0xF0)>>4;
    return gray;
}

void Fea::setGLCMfeatures(double *glcm, int index, double glcmMatrix[][GLCM_CLASS])
{
    for(int i=0;i<GLCM_CLASS;i++)
        for(int j=0;j<GLCM_CLASS;j++)
        {
            // entropy
            if(glcmMatrix[i][j]>0)
                glcm[index*4] -= glcmMatrix[i][j]*log10(glcmMatrix[i][j]);
            // energy
            glcm[index*4+1] += glcmMatrix[i][j]*glcmMatrix[i][j];
            // contrast
            glcm[index*4+2] += (i-j)*(i-j)*glcmMatrix[i][j];
            // homogenity
            glcm[index*4+3] += 1.0/(1+(i-j)*(i-j))*glcmMatrix[i][j];
        }
}

void Fea::deComposeMV(std::vector<glm::vec3> &v_eye, std::vector<glm::vec3> &v_center, std::vector<glm::vec3> &v_up)
{
    for(int i=0;i<m_modelList.size();i++)
    {
        glm::vec3 t = glm::vec3(m_modelList[i][3]);
        glm::mat3 R = glm::mat3(m_modelList[i]);

        glm::vec3 eye = - glm::transpose(R) * t;
        glm::vec3 center = glm::normalize(glm::transpose(R) * glm::vec3(0.f,0.f,-1.f)) + eye;
        glm::vec3 up = glm::normalize(glm::transpose(R) * glm::vec3(0.f,1.f,0.f));

        v_eye.push_back(eye);
        v_center.push_back(center);
        v_up.push_back(up);
    }
}

void Fea::clear()
{
    image.release();

    for(int i=0;i<contour.size();i++)
        contour[i].clear();
    contour.clear();

    mask.release();

    image2D.release();

    gray.release();

    fea2D.clear();

    fea3D.clear();

    fea2DName.clear();

    fea3DName.clear();
}

void Fea::exportSBM(QString file)
{
    int MAX_X_LEN = 16;
    int MAX_Y_LEN = 64;

    glm::mat4 mv;
    // villa5 model
//        glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 5.0f, 80.f);
    // villa4 model
//    glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 400.0f, 1200.f);
    // villa3 model
//    glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 1250.0f, 8000.f);
    // villa2 model
//    glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 5000.0f, 25000.f);
    // villa model
//    glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 1000.0f, 2000.f);
//    float angle_x = 2.0*glm::pi<float>()/MAX_LEN;
//    villa6 model
    glm::mat4 proj = glm::perspective(glm::pi<float>() / 3, 4.0f / 3.0f, 4000.0f, 20000.f);
//    float angle_x = glm::pi<float>() / 12.0 / MAX_LEN;
    // villa7
//    float angle_x = glm::pi<float>() / 36.0 / MAX_X_LEN;
    // villa7_1
    float angle_x = glm::pi<float>() / 2.0 / MAX_X_LEN;
    float angle_z = 2.0*glm::pi<float>()/MAX_Y_LEN;
//    float angle_z = glm::pi<float>() / MAX_LEN;
    // zb Model parameters
//    glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,0.f,80.f),
//                           glm::vec3(0.f,5.0f,0.0f),
//                           glm::vec3(0.f,1.f,0.f));
    // villa2 Model
//    glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,1500.f,18000.f),
//                           glm::vec3(0.f,3000.0f,0.0f),
//                           glm::vec3(0.f,1.f,0.f));
    // villa Model
//    glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,0.f,1500.f),
//                           glm::vec3(0.f,200.0f,0.0f),
//                           glm::vec3(0.f,1.f,0.f));

        // villa3 Model
//        glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,800.f,6000.f),
//                               glm::vec3(0.f,500.0f,0.0f),
//                               glm::vec3(0.f,1.f,0.f));

//     villa4 Model
//    glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,200.f,800.f),
//                           glm::vec3(0.f,200.0f,0.0f),
//                           glm::vec3(0.f,1.f,0.f));

// villa5 Model
//    glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,5.f,30.f),
//                           glm::vec3(0.f,5.0f,0.0f),
//                           glm::vec3(0.f,1.f,0.f));

// villa6 Model
        glm::mat4 m_camera = glm::lookAt(glm::vec3(0.f,-800.f,11000.f),
                               glm::vec3(0.f,0.0f,0.0f),
                               glm::vec3(0.f,1.f,0.f));


    std::ofstream fout(file.toStdString().c_str());
    int ind = 0;
    for(int i=0;i<MAX_X_LEN;i++)
    {
        for(int j=0;j<MAX_Y_LEN;j++)
        {
            // rotate with x axis
            float anglex = angle_x * i;
            // rotate with y axis
            float anglez = angle_z * j - glm::pi<float>() / 2.0;
            glm::mat4 rotateX = glm::rotate(glm::mat4(1.f),anglex,glm::vec3(1.0,0.0,0.0));
            glm::mat4 rotateZ = glm::rotate(glm::mat4(1.f),anglez,glm::vec3(0.0,1.0,0.0));
            mv = m_camera * rotateX * rotateZ;
            // print out
            fout << "img" ;
            fout.width(4);
            fout.fill('0');
//            fout << i*MAX_LEN + j << ".jpg"<<std::endl;
            fout << ind++ << ".jpg" << std::endl;

            // mv matrix
            for(int k1 = 0;k1 < 4;k1++)
            {
                for(int k2 = 0;k2 < 4;k2++)
                    fout << mv[k2][k1] << " ";
                fout << std::endl;
            }
            // proj matrix
            for(int k1 = 0;k1 < 4;k1++)
            {
                for(int k2 = 0;k2 < 4;k2++)
                    fout << proj[k2][k1] << " ";
                fout << std::endl;
            }
        }
    }
    std::cout <<MAX_Y_LEN<<" export done" << std::endl;
}

float Fea::floatAbs(float num)
{
    return num < 0 ? -num : num;
}
