#include "fea.hh"
#include "meancurvature.hh"
#include "gausscurvature.hh"
#include <QtDebug>
#include "predefine.h"


Fea::Fea(QString fileName, QString path)
{
    this->path = path;

    // the number of feature is 12
    feaArray = new double[12];

    memset(feaArray,0,sizeof(double)*12);
    // read in
    exImporter = new ExternalImporter<MyMesh>();

    if(!exImporter->read_mesh(mesh,fileName.toStdString().c_str()))
    {
        std::cerr << "Error: Cannot read mesh from "<<std::endl;
        return ;
    }  

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
    QString mmFile = fileName;
    int index = mmFile.lastIndexOf('.');
    mmFile.replace(index,6,".mm");
    setMMPara(mmFile);
//    ui->lmatrixPath->setText(QString("path: ").append(mmFile));
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

    setFeature();
}

void Fea::setFeature()
{

#ifdef CHECK
    glm::mat4 m_model_tmp;

    glm::mat4 m_view_tmp;

    MeanCurvature<MyMesh> a(mesh);
    GaussCurvature<MyMesh> b(mesh);

#else
    // for compute
    render->setMeshSaliencyPara(exImporter);

    std::vector<MeanCurvature<MyMesh>> a;
    std::vector<GaussCurvature<MyMesh>> b;
    for(int i=0;i<render->p_vecMesh.size();i++)
    {
        MeanCurvature<MyMesh> tmpMean(render->p_vecMesh[i]);
        GaussCurvature<MyMesh> tmpGauss(render->p_vecMesh[i]);
        a.push_back(tmpMean);
        b.push_back(tmpGauss);
    }

#endif

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
//            render->showImage();

//            qDebug()<<" set feature "<<path<<endl;
//            std::cout<<path.toStdString()<<std::endl;

//            render->storeImage(path,QString::number(t_case));

//            qDebug()<<" store Image ok "<<endl;

            setMat(render->p_img,render->p_width,render->p_height);

            setProjectArea();

            setVisSurfaceArea(render->p_vertices,render->p_VisibleFaces);

            setViewpointEntropy2(render->p_verticesMvp,render->p_VisibleFaces);

            setSilhouetteLength();

//            setSilhouetteCE();

            setMaxDepth(render->p_img,render->p_height*render->p_width);

            setDepthDistribute(render->p_img,render->p_height*render->p_width);

#ifdef CHECK

            setMeanCurvature(a,render->p_isVertexVisible);

            setGaussianCurvature(b,render->p_isVertexVisible);

//            setMeshSaliency(a,render->p_vertices,render->p_isVertexVisible);

            setAbovePreference(m_abv,m_model_tmp,m_view_tmp);

#else

            setMeanCurvature(a,render->p_isVertexVisible,render->p_indiceArray);

            setGaussianCurvature(b,render->p_isVertexVisible,render->p_indiceArray);

            setMeshSaliency(a,render->p_vertices,render->p_indiceArray);

            // setAbovePreference 有问题,compute缺少参数
            // setAbovePreference(m_abv,m_model_tmp,m_view_tmp);

#endif

//            clear();

        }

        break;
        printOut();

    }

}

void Fea::setMMPara(QString mmFile)
{
    this->mmPath = mmFile;

    output = mmFile;

    int pos = output.lastIndexOf('.');
    output.replace(pos,7,".3df");

    std::cout<< "output " << output.toStdString() << std::endl;

//    set_tCase();

//    setFilenameList_mvpMatrix(matrixPath);

//    NUM = fileName.count();

//    for(int i = 0; i<NUM ; i++)
//        std::cout<<fileName.at(i).toStdString()<<std::endl;
#ifdef CHECK
    // for check there are two matrixs
    // the first one is the view matrix
    // the second one is the model matrix and also is the abv matrix
    // the last two para is from to
    freopen(mmPath.toStdString().c_str(),"r",stdin);
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
    // for compute there is one matirx is abv matrix
    // and the last two para is from to
    float tmp;
    for(int i=0;i<16;i++)
    {
        scanf("%f",&tmp);
        m_abv[i/4][i%4] = tmp;
    }
    scanf("%d %d",&t_case,&NUM);
#endif
    std::cout<<t_case<<" "<<NUM<<std::endl;

}


Fea::~Fea()
{

}

void Fea::setMat(float *img, int width, int height)
{
//    image.release();
    cv::Mat image0 = cv::Mat(width,height,CV_32FC1,img);
    image0.convertTo(image,CV_8UC1,255.0);
    // release memory
    image0.release();
}

void Fea::setProjectArea()
{
    feaArray[0] = 0.0;
    cv::Mat img = cv::Mat(image.rows,image.cols,CV_8UC1);
    if(image.channels()==3)
    {
        for(int i=0;i<image.rows;i++)
            for(int j=0;j<image.cols;j++)
                if(image.at<uchar>(i,j,1)!=255
                   || image.at<uchar>(i,j,2)!=255
                   || image.at<uchar>(i,j,3)!=255)
                feaArray[0]++;
    }
    else
    {

        for(int i=0;i<image.rows;i++)
            for(int j=0;j<image.cols;j++)
            {
                if(image.at<uchar>(i,j)!=255)
                {
                    feaArray[0]++;
                    img.at<uchar>(i,j) = 255;
                }
                else
                {
                    img.at<uchar>(i,j) = 0;
                }
            }

    }

    QString projPath = path + QString("proj/").append(QString::number(t_case)).append(QString(".jpg"));

    cvSaveImage(projPath.toStdString().c_str(),&(IplImage(img)));

    img.release();
    std::cout<<"fea projectArea "<<feaArray[0]<<std::endl;

}

void Fea::setVisSurfaceArea(std::vector<GLfloat> &vertex,
                             std::vector<GLuint> &face)
{

    feaArray[1] = 0;
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
        feaArray[1] += getArea3D(&p1,&p2,&p3);
    }

    std::cout<<"fea visSurfaceArea "<< feaArray[1]<<std::endl;
    // used for test
}

void Fea::setViewpointEntropy2(std::vector<GLfloat> &vertex, std::vector<GLuint> &face)
{
    feaArray[2] = 0.0;
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
            feaArray[2] += area/totalArea * log2(area/totalArea);
        else
            qDebug()<<"viewpoint "<<area<<endl;
    }
    // background
    if((feaArray[0] - totalArea) > 0)
        feaArray[2] += (totalArea - feaArray[0])/totalArea * log2((totalArea - feaArray[0])/totalArea);

    feaArray[2] = - feaArray[2];
    std::cout<<"fea viewpointEntropy "<<feaArray[2]<<std::endl;
}

void Fea::setViewpointEntropy(std::vector<GLfloat> &vertex, std::vector<GLuint> &face)
{
//    double hist[15];
    double *hist = new double[NumHistViewEntropy];
//    还有此处，写成 double hist[15]; memest(hist,0,sizeof(hist));也会出错！
    memset(hist,0,sizeof(double)*NumHistViewEntropy);
    double *area = new double[face.size()/3];
    double min = 1e10;
    double max = -1.0;
    feaArray[2] = 0.0;
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
            feaArray[2] += hist[i]*log2(hist[i]);
//    NND绝对的未解之谜！加了下面一句话会报错！
    delete []hist;
    feaArray[2] = - feaArray[2];

    std::cout<<"fea viewpointEntropy "<<feaArray[2]<<std::endl;
//    delete []hist;
    delete []area;
/*
    freopen("e:/matlab/vpe.txt","w",stdout);
    for(int i=0;i<vertex.size();i+=3)
        printf("%f %f %f\n",vertex[i],vertex[i+1],vertex[i+2]);
    fclose(stdout);
*/

}

void Fea::setSilhouetteLength()
{
//    feaArray[3] = 0.0;
////    ref http://blog.csdn.net/augusdi/article/details/9000893
//    IplImage *tmpImage =
//            cvCreateImage(cvSize(image.cols,image.rows),
//                          8,1);
//    if(image.channels()==3)
//        cv::cvtColor(image,image,CV_BGR2GRAY);

//    tmpImage->imageData = (char*)image.data;
//    cvThreshold(tmpImage,tmpImage,250,255,CV_THRESH_BINARY_INV);
////    cvShowImage("tmpimage",tmpImage);
//    IplImage *img_tmp =
//            cvCreateImage(cvGetSize(tmpImage),8,1);
//    img_tmp = cvCloneImage(tmpImage);

//    mem_storage = cvCreateMemStorage(0);

//    contour = NULL;
//    cvFindContours(
//                img_tmp,
//                mem_storage,
//                &contour,
//                sizeof(CvContour),
//                CV_RETR_EXTERNAL
//                );
//    cvZero(img_tmp);
//    cvDrawContours(
//                img_tmp,
//                contour,
//                cvScalar(100),
//                cvScalar(100),
//                1);
////    cvShowImage("image",img_tmp);
////    ref http://blog.csdn.net/fdl19881/article/details/6730112
//    if(contour)
//        feaArray[3] = cvArcLength(contour);
//    else
//        feaArray[3] = 0.0;
//    std::cout<<"fea silhouetteLength "<<feaArray[3]<<std::endl;
//    cvReleaseImage(&tmpImage);
//    cvReleaseImage(&img_tmp);


    feaArray[3] = 0.0;
    // ref http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
    cv::Mat gray = image.clone();

    // ref http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
    // 这个一定要二值化，图像本身就基本都是白色，直接提取轮廓是拿不到结果的
    cv::threshold( gray, gray, 254, 255.0,3 );

//    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(gray,contour,hierarchy,CV_RETR_EXTERNAL ,CV_CHAIN_APPROX_NONE );

    if(contour.size())
        feaArray[3] = cv::arcLength(contour,true);
    else
        feaArray[3] = 0.0;
    std::cout<<"fea silhouetteLength "<<feaArray[3]<<std::endl;

    std::vector<cv::Vec4i>().swap(hierarchy);

// see contour result

    cv::Mat drawing = cv::Mat::zeros(gray.size(),CV_8UC3);
//    for(int i=0;i<contours.size();i++)
    for(int i=0 ; i<1 ; i++)
    {
        cv::Scalar color = cv::Scalar(255,255,255);
        cv::drawContours(drawing,contour,i,color,2,8,hierarchy,0,cv::Point());
    }

    cv::namedWindow("contours");
    cv::imshow("contours",drawing);

}
/*
void Fea::setSilhouetteCE()
{
    feaArray[4] = 0.0;
    feaArray[5] = 0.0;
    double curva = 0;
    double dis = 0.0;

//    example
//    ghabcdefghabcde
//     ^  ->  ^
//    gha -> hab -> abc
    if(contour)
    for(int i=0;i<contour->total;i++)
    {
        CvPoint *a0 = CV_GET_SEQ_ELEM(CvPoint,contour,i-2);
        CvPoint *b0 = CV_GET_SEQ_ELEM(CvPoint,contour,i-1);
        CvPoint *c0 = CV_GET_SEQ_ELEM(CvPoint,contour,i);
        CvPoint2D64f a = cvPoint2D64f((double)a0->x,(double)a0->y);
        CvPoint2D64f b = cvPoint2D64f((double)b0->x,(double)b0->y);
        CvPoint2D64f c = cvPoint2D64f((double)c0->x,(double)c0->y);

        std::vector<cv::Point2d> points;
        points.push_back(cv::Point2d(a.x, a.y));
        points.push_back(cv::Point2d(b.x, b.y));
        points.push_back(cv::Point2d(c.x, c.y));


//        if(getCurvature(&a,&b,&c,curva))
//        {
////            std::cout << curva << std::endl;
            dis = getDis2D(&a,&b) + getDis2D(&b,&c);
//            feaArray[4] += abs(curva) * dis;
//            feaArray[5] += curva*curva * dis;
//        }

        double curvab = getContourCurvature(points,1);
        if (std::isnan(curvab)) {
//            qDebug()<<a.x<<" "<<a.y<<endl;
//            qDebug()<<b.x<<" "<<b.y<<endl;
//            qDebug()<<c.x<<" "<<c.y<<endl;
//            assert(0);
        }
        else
        {
            feaArray[4] += abs(curvab);
            feaArray[5] += curvab*curvab;
        }

//        qDebug()<<"curvature a"<<curva<<" "<<abs(curvab)<< " "<<abs(curvab) - abs(curva)<<endl;
    }

    std::cout<<"fea silhouetteCurvature "<<feaArray[4]<<std::endl;
    std::cout<<"fea silhouetteCurvatureExtrema "<<feaArray[5]<<std::endl;
}
*/
void Fea::setMaxDepth(float *array,int len)
{
    feaArray[6] = -10.0;
    for(int i=0;i<len;i++)
        if(array[i] < 1.0)
            feaArray[6] = feaArray[6] > array[i] ? feaArray[6] : array[i];
    std::cout<<"fea maxDepth "<<feaArray[6]<<std::endl;
}

void Fea::setDepthDistribute(float *zBuffer, int num)
{
    feaArray[7] = 0.0;
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
    if(step)
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
        feaArray[7] += hist[i]*hist[i]*step;
    feaArray[7] = 1 - feaArray[7];

    std::cout<<"fea depthDistriubute "<<feaArray[7]<<std::endl;
    delete []hist;
//    qDebug()<<"depth distribute"<<endl;
}

void Fea::setMeanCurvature(MeanCurvature<MyMesh> &a, std::vector<bool> &isVertexVisible)
{
    feaArray[8] = 0.0;
    feaArray[8] = a.getMeanCurvature(isVertexVisible);
    if(feaArray[0])
        feaArray[8] /= feaArray[0];
    std::cout<<"fea meanCurvature "<<feaArray[8]<<std::endl;
}

void Fea::setMeanCurvature(std::vector<MeanCurvature<MyMesh>> &a,
                           std::vector<bool> &isVertexVisible,
                           std::vector<std::vector<int>> &indiceArray)
{
//    printf("vecMesh....%d\n",vecMesh.size());
//    printf("indiceArray....%d\n",indiceArray.size());
    feaArray[8] = 0.0;
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
//        MeanCurvature<MyMesh> a(vecMesh[i]);
        feaArray[8] += a[i].getMeanCurvature(isVerVis);
    }

    if(feaArray[0])
        feaArray[8] /= feaArray[0];
    std::cout<<"fea meanCurvature "<<feaArray[8]<<std::endl;
}

void Fea::setGaussianCurvature(GaussCurvature<MyMesh> &mesh,
                               std::vector<bool> &isVertexVisible)
{
    feaArray[9] = 0.0;
//    GaussCurvature<MyMesh> a(mesh);
    feaArray[9] = mesh.getGaussianCurvature(isVertexVisible);
    if(feaArray[0])
        feaArray[9] /= feaArray[0];
    std::cout<<"fea gaussianCurvature "<<feaArray[9]<<std::endl;
}

void Fea::setGaussianCurvature(std::vector<GaussCurvature<MyMesh>> &a,
                               std::vector<bool> &isVertexVisible,
                               std::vector<std::vector<int>> &indiceArray)
{
    feaArray[9] = 0.0;
    for(int i=0;i<a.size();i++)
    {
        std::vector<bool> isVerVis;

        std::set<int> verIndice;
        for(int j=0;j<indiceArray[i].size();j++)
            verIndice.insert(indiceArray[i][j]);
        std::set<int>::iterator it = verIndice.begin();
        for(;it!=verIndice.end();it++)
            isVerVis.push_back(isVertexVisible[*it]);

        feaArray[9] += a[i].getGaussianCurvature(isVerVis);
    }
    if(feaArray[0])
    feaArray[9] /= feaArray[0];
    std::cout<<"fea gaussianCurvature "<<feaArray[9]<<std::endl;
}

void Fea::setMeshSaliency(MeanCurvature<MyMesh> &a, std::vector<GLfloat> &vertex, std::vector<bool> &isVertexVisible)
{
    feaArray[10] = 0.0;
    double length = getDiagonalLength(vertex);
    std::vector<double> meanCurvature;
    double *nearDis = new double[vertex.size()/3];
//    MeanCurvature<MyMesh> a(mesh);
    double sigma[5] = {0.003*2.0,0.003*3.0,0.003*4.0,0.003*5.0,0.003*6.0};
    std::vector<double> meshSaliencyMiddle[5];
    double localMax[5];
    double gaussWeightedVal1,gaussWeightedVal2;
    a.setMeanCurvature(meanCurvature);
    for(int j=0;j<5;j++)
    {
        localMax[j] = 0.0;
        for(int i=0;i<vertex.size();i+=3)
        {
            setNearDisMeshSaliency(vertex,i,length,sigma[j],nearDis);
            gaussWeightedVal1 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]);
            gaussWeightedVal2 = getGaussWeightedVal(meanCurvature[i/3],nearDis,vertex.size()/3,sigma[j]*2.0);
            meshSaliencyMiddle[j].push_back(abs(gaussWeightedVal1 - gaussWeightedVal2));
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
            feaArray[10] += meshSaliencyMiddle[0][i];
//    std::cout<<"fea meshSaliency ";
    printf("fea meshSaliency %e\n",feaArray[10]);

    delete []nearDis;
}

void Fea::setMeshSaliency(int t_case,// for debug can be used to output the mesh
                          std::vector<GLfloat> &vertex,
                          std::vector<bool> &isVertexVisible,
                          std::vector<MeanCurvature<MyMesh>> &a,
                          std::vector<std::vector<int>> &indiceArray)
{


    feaArray[10] = 0.0;
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
            meshSaliencyMiddle[j].push_back(abs(gaussWeightedVal1 - gaussWeightedVal2));
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
            feaArray[10] += meshSaliencyMiddle[0][i];
    std::cout<<"fea meshSaliency "<<feaArray[10]<<std::endl;

    delete []nearDis;
    delete []meanCurvature;
}

void Fea::setAbovePreference(double theta)
{
    this->feaArray[11] = 0.0;
    double pi = asin(1.0)*2.0;
    feaArray[11] = exp(-(theta - pi/8.0*3.0)*(theta - pi/8.0*3.0)
                          / pi/4.0*pi/4.0);
}

void Fea::setAbovePreference(glm::mat4 &model2, glm::mat4 &model,glm::mat4 &view)
{

//    int pos = filename.lastIndexOf('.');
//    filename.replace(pos,6,".mm");
//    filename.append(QString(".mm"));
//    FILE *fp = freopen(filename.toStdString().c_str(),"r",stdin);
//    if(fp)
//    {
//        glm::mat4 model2;
//        double tmp;
//        for(int i=0;i<4;i++)
//            for(int j=0;j<4;j++)
//            {
//                scanf("%lf",&tmp);
//                model2[i][j] = tmp;
//            }
//        model2 = glm::transpose(model2);
        glm::vec4 z = glm::vec4(0.0,0.0,1.0,0.0);
        glm::vec4 yyy = model*model2*z;
    //    the theta between yyy and (0,1,0,1)
//        qDebug()<<"........."<<endl;
//        qDebug()<<yyy.x<<" "<<yyy.y<<" "<<yyy.z<<" "<<yyy.w<<endl;
        // ref http://stackoverflow.com/questions/21830340/understanding-glmlookat
        // I need center to eye // center - eye
        glm::vec4 lookAxis = glm::vec4(-view[0][2],-view[1][2],-view[2][2],0.f);

        float norm_yyy = glm::dot(yyy,yyy);

        float norm_lookAxis = glm::dot(lookAxis,lookAxis);

        float dot = glm::dot(yyy,lookAxis);

        double cosTheta = dot / sqrt(norm_yyy) / sqrt(norm_lookAxis);

        double theta = acos(cosTheta);

        setAbovePreference(theta);

    std::cout<<"abovePreference "<<feaArray[11]<<std::endl;
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
    freopen(matrixPath.toStdString().c_str(),"r",stdin);
    QString tmp;
    char tmpss[200];
    float tmpNum;
    while(scanf("%s",tmpss)!=EOF)
    {
        QString tmpPath = path;
        tmp = QDir::cleanPath(tmpPath.append(QString(tmpss)));
        fileName.push_back(tmp);
#ifdef NoProjection
        glm::mat4 m,v;
        for(int i=0;i<16;i++)
        {
            scanf("%f",&tmpNum);
            m[i%4][i/4] = tmpNum;
        }
        this->m_modelList.push_back(m);
        this->m_viewList.push_back(v);

        glm::mat4 p = glm::perspective(glm::pi<float>() / 2, 1.f, 0.1f, 100.f);
        this->m_projectionList.push_back(p);
#else
        glm::mat4 m,v,p;
        for(int i=0;i<16;i++)
        {
            scanf("%lf",&tmpNum);
            m[i/4][i%4] = tmpNum;
        }
        for(int i=0;i<16;i++)
        {
            scanf("%lf",&tmpNum);
            p[i/4][i%4] = tmpNum;
        }
        this->model.push_back(m);
        this->view.pish_back(v);
        this->projection.push_back(p);
#endif
    }
}

void Fea::printOut()
{
    freopen(output.toStdString().c_str(),"a+",stdout);
#ifdef CHECK
    // output the order number
    printf("%s\n",QString::number(t_case).toStdString().c_str());
#else
    // output the fileName
    printf("%s\n",fileName[i].toStdString().c_str());
#endif

    for(int i=0;i<12;i++)
    {
        if(i==8 || i== 9 || i == 10)
            printf("%e ",feaArray[i]);
        else
            printf("%lf ",feaArray[i]);
    }
    printf("\n");
    fclose(stdout);
#ifdef CHECK
    freopen(mmPath.toStdString().c_str(),"w",stdout);
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
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
            printf("%f ",m_abv[i][j]);
        printf("\n");
    }

    printf("%d %d\n",t_case+1,NUM);


#endif

    freopen("CON","a+",stdout);
    image.release();
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

void Fea::showImage()
{
    render->showImage();
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

void Fea::clear()
{
    qDebug() << "clear mem_storage "<<mem_storage<<endl;
//    qDebug() << "clear contour "<<contour<<endl;
    qDebug() << "clear image "<<&image<<endl;

    image.release();

    qDebug() << "clear hi 1 "<<endl;
    // ref http://stackoverflow.com/questions/5951292/how-do-you-delete-a-cvseq-in-opencv
    // Clear the memory storage which was used before

//    cvClearSeq(contour);

    cvClearMemStorage(mem_storage);

    qDebug() << "clear hi 2 "<<endl;
    // Release memory
    cvReleaseMemStorage(&mem_storage);
    qDebug() << "clear hi 3 "<<endl;

}
