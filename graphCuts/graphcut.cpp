#include "graphcut.h"

GraphCut* graphCutInstance;
void on_mouse(int event, int x, int y, int flags, void* param);

GraphCut::GraphCut()
{
    image = cv::Mat(0,0,CV_32F);
    mask = cv::Mat(0,0,CV_32F);
    foregroundMask = cv::Mat(0,0,CV_32F);
    backgroundMask = cv::Mat(0,0,CV_32F);
    numClusters = 8;
    kernelSize = 5;
    opTimes = 20;
}

GraphCut::~GraphCut()
{

}

void GraphCut::cut()
{
    beta = 5.0;
    alpha =10000.0;
    cv::Vec3b pix = 0;
    double weightSource;
    double weightSink;
    // read in first
    if(image.rows == 0  || image.cols == 0 )
    {
        printf("read in error!\n");
        return ;
    }
    typedef Graph<int ,int ,int> GraphType;

    int numEdges = (image.cols - 1) * image.rows;
    numEdges += image.cols * (image.rows - 1);

    GraphType *g = new GraphType(image.rows*image.cols,numEdges);

    for(int i=0;i<image.cols*image.rows;i++)
        g->add_node();

//    FILE *fp1,*fp2;
//    fp1 = fopen("D:/fore.txt","w");
//    fp2 = fopen("D:/back.txt","w");
    // add tweight
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            pix = image.at<cv::Vec3b>(i,j);
            weightSink = computeGMM((double)pix[0],(double)pix[1],(double)pix[2]);
            weightSource = computeGMM2((double)pix[0],(double)pix[1],(double)pix[2]);

//            fprintf(fp1,"%e\n",weightSink);
//            fprintf(fp2,"%e\n",weightSource);

            weightSink = alpha * -log((long double)(weightSink));
            weightSource = alpha * -log((long double)(weightSource));
            g->add_tweights(i * image.cols + j, weightSource,weightSink);
        }
    }

//    fclose(fp1);
//    fclose(fp2);

    // add edge
    // up to down && down to up
    for(int i=0;i<image.cols;i++)
    {
        for(int j=1;j<image.rows;j++)
        {
            int weight = abs(image.at<cv::Vec3b>(j,i)[0] - image.at<cv::Vec3b>(j-1,i)[0]);
            weight += abs(image.at<cv::Vec3b>(j,i)[1] - image.at<cv::Vec3b>(j-1,i)[1]);
            weight += abs(image.at<cv::Vec3b>(j,i)[2] - image.at<cv::Vec3b>(j-1,i)[2]);

            cv::Vec3d a1 = image.at<cv::Vec3b>(j,i);
            cv::Vec3d a2 = image.at<cv::Vec3b>(j-1,i);
            double theta = a1.dot(a2);
            theta /= sqrt(a1.dot(a1) * a2.dot(a2));

            g->add_edge(j*image.cols+i,(j-1)*image.cols+i,beta * weight,beta * weight);
        }
    }
    printf("ok ... 6\n");
    // left to right && right to left
    for(int i=0;i<image.rows;i++)
    {
        for(int j=1;j<image.cols;j++)
        {
            int weight = abs(image.at<cv::Vec3b>(i,j)[0] - image.at<cv::Vec3b>(i,j-1)[0]);
            weight += abs(image.at<cv::Vec3b>(i,j)[1] - image.at<cv::Vec3b>(i,j-1)[1]);
            weight += abs(image.at<cv::Vec3b>(i,j)[2] - image.at<cv::Vec3b>(i,j-1)[2]);

            cv::Vec3d a1 = image.at<cv::Vec3b>(i,j);
            cv::Vec3d a2 = image.at<cv::Vec3b>(i,j-1);
            double theta = a1.dot(a2);
            theta /= sqrt(a1.dot(a1) * a2.dot(a2));

            g->add_edge(i*image.cols+j,i*image.cols+j-1,beta * weight,beta * weight);
        }
    }
    printf("ok ... 7\n");
    int flow =  g->maxflow();

//    printf("ok ... 8\n");

    cv::Mat result = cv::Mat(image.rows,image.cols,CV_8UC3);

    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            int id = i*image.cols + j;
            if(g->what_segment(id) == GraphType::SOURCE)
            {
                result.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i,j)[0];
                result.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i,j)[1];
                result.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i,j)[2];
            }
            else
            {

                result.at<cv::Vec3b>(i,j)[0] = 0;
                result.at<cv::Vec3b>(i,j)[1] = 0;
                result.at<cv::Vec3b>(i,j)[2] = 0;
            }
        }
    }

//    cv::Mat maskImg = cv::Mat(image.rows,image.cols,CV_8UC3);
//    for(int i=0;i<image.rows;i++)
//    {
//        for(int j=0;j<image.cols;j++)
//        {
//            if(gray.at<uchar>(i,j) > 128)
//            {
//                maskImg.at<cv::Vec3b>(i,j)[0] = 0;
//                maskImg.at<cv::Vec3b>(i,j)[1] = 0;
//                maskImg.at<cv::Vec3b>(i,j)[2] = 0;
//            }
//            else
//            {
//                maskImg.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i,j)[0];
//                maskImg.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i,j)[1];
//                maskImg.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i,j)[2];
//            }
//        }
//    }

    delete g;

    cv::namedWindow("result");
    cv::imshow("result",result);

//    cv::namedWindow("mask");
//    cv::imshow("mask",mask);
    //    fclose(stdout);
}

void GraphCut::cutting()
{
    cv::Mat srcImage;
    image.copyTo(srcImage);
    cv::namedWindow("pstrWindowsMouseDrawTitle");
    cv::imshow("pstrWindowsMouseDrawTitle",srcImage);
    cv::setMouseCallback("pstrWindowsMouseDrawTitle",on_mouse,(void*)&srcImage);

    int c;
    do{
            c = cv::waitKey();
            switch ((char)c)
            {
            case 'r':
                //pSrcImage.setTo(Scalar(255, 255, 255));
                //imshow(pstrWindowsMouseDrawTitle, pSrcImage);
                image.copyTo(srcImage);
                foregroundMask.setTo(0);
                backgroundMask.setTo(0);
                cv::imshow("pstrWindowsMouseDrawTitle", srcImage);
                break;

            case 'm':
                makeMask();
                cv::namedWindow("mask");
                cv::imshow("mask",foregroundMask);
                break;
            case 'c':
                cut();
                break;
            }
        }while (c > 0 && c != 27);
}

void GraphCut::thinnig()
{

    cv::namedWindow("mask");
    cv::imshow("mask",mask);

    cv::cvtColor(mask,gray,CV_BGR2GRAY);



    cv::threshold(gray,gray,104,255,CV_THRESH_BINARY);

    gray /= 255;

//    cv::namedWindow("mask");
//    cv::imshow("mask",gray);

    cv::Mat prev = cv::Mat::zeros(gray.size(), CV_8UC1);

    cv::Mat diff;

        do {
            thinningIteration(gray, 0);
            thinningIteration(gray, 1);
            cv::absdiff(gray, prev, diff);
            gray.copyTo(prev);
        }
        while (cv::countNonZero(diff) > 0);

        gray *= 255;

        cv::namedWindow("thin");
        cv::imshow("thin",gray);

}

void GraphCut::readIn(QString file,QString fileMask)
{
    image = cv::imread(file.toStdString().c_str());

    mask = cv::imread(fileMask.toStdString().c_str(),0);

//    printf("read in ok\n");

//    cv::namedWindow("image");
//    cv::imshow("image",image);
}

void GraphCut::thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void GraphCut::makeMask(int fromRow, int fromCol, int toRow, int toCol)
{
    gray = cv::Mat(image.rows,image.cols,CV_8UC1,cv::Scalar(0));
    for(int i=fromRow;i<toRow;i++)
        for(int j=fromCol;j<toCol;j++)
            gray.at<uchar>(i,j) = 255;

    cv::namedWindow("gray");
    cv::imshow("gray",gray);

    trainData1 = new double[(toRow-fromRow)*(toCol - fromCol)*3];
//    trainData[0] = new float[(toRow-fromRow)*(toCol - fromCol)];
//    trainData[1] = new float[(toRow-fromRow)*(toCol - fromCol)];
//    trainData[2] = new float[(toRow-fromRow)*(toCol - fromCol)];
    int count = 0;
    for(int i=fromRow;i<toRow;i++)
    {
        for(int j = fromCol;j<toCol;j++)
        {
            trainData1[count++] = image.at<cv::Vec3b>(i,j)[0];
            trainData1[count++] = image.at<cv::Vec3b>(i,j)[1];
            trainData1[count++] = image.at<cv::Vec3b>(i,j)[2];
//            trainData[count] = new float[3];
//            trainData[count][1] = image.at<cv::Vec3b>(i,j)[0];
//            trainData[count][2] = image.at<cv::Vec3b>(i,j)[1];
//            trainData[count][3] = image.at<cv::Vec3b>(i,j)[2];

//            printf("%f %f %f\n",trainData[0][count],trainData[1][count],trainData[2][count]);
//            count++;
        }
    }

//    count  = 0;
//    data = new double[image.rows*image.cols*3];
//    for(int i=0;i<image.rows;i++)
//    {
//        for(int j=0;j<image.cols;j++)
//        {
//            data[count++] = image.at<cv::Vec3b>(i,j)[0];
//            data[count++] = image.at<cv::Vec3b>(i,j)[1];
//            data[count++] = image.at<cv::Vec3b>(i,j)[2];
//        }
//    }

    //ref http://www.vlfeat.org/api/gmm.html
    VlGMM *gmm;
//    double loglikelihood ;
    int dimension = 3;
    numClusters = 5;
    // create a new instance of a GMM object for float data
    gmm = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
    // set the maximum number of EM iterations to 100
    vl_gmm_set_max_num_iterations (gmm, 100) ;
    // set the initialization to random selection
    vl_gmm_set_initialization (gmm,VlGMMRand);
    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster (gmm, trainData1, (toRow-fromRow)*(toCol - fromCol));
    // get the means, covariances, and priors of the GMM
    means1 = (double*)vl_gmm_get_means(gmm);
    covariances1 = (double*)vl_gmm_get_covariances(gmm);
    priors1 = (double*)vl_gmm_get_priors(gmm);
    // get loglikelihood of the estimated GMM
//    loglikelihood = vl_gmm_get_loglikelihood(gmm) ;
    // get the soft assignments of the data points to each cluster
//    posteriors = (double*)vl_gmm_get_posteriors(gmm) ;

    printf("ok ... \n");

//    for(int i=0;i<numClusters;i++)
//        printf("mean %lf %lf %lf\n",means[i*3],means[i*3+1],means[i*3+2]);
//    for(int i=0;i<numClusters;i++)
//        printf("covariances %lf %lf %lf\n",covariances[3*i],covariances[3*i+1],covariances[3*i+2]);
//    for(int i=0;i<numClusters;i++)
//        printf("priors %f\n",priors[i]);


}

void GraphCut::makeMask()
{
    int count = 0;
    int num0 = 0;
    int num1 = 0;
    for(int i=0;i<foregroundMask.rows;i++)
    {
        for(int j=0;j<foregroundMask.cols;j++)
        {
            if(foregroundMask.at<cv::Vec3b>(i,j)[2] == 255 &&
                    foregroundMask.at<cv::Vec3b>(i,j)[1] == 0 &&
                    foregroundMask.at<cv::Vec3b>(i,j)[0] == 0)
                count ++;
        }
    }

    for(int i=0;i<backgroundMask.rows;i++)
    {
        for(int j=0;j<backgroundMask.cols;j++)
        {
            if(backgroundMask.at<cv::Vec3b>(i,j)[2] == 0 &&
                    backgroundMask.at<cv::Vec3b>(i,j)[1] == 0 &&
                    backgroundMask.at<cv::Vec3b>(i,j)[0] == 255)
                num1++;
        }
    }

    printf("foregroundMask done %d\n",num1);

//    cv::namedWindow("background");
//    cv::imshow("background",backgroundMask);

    num0 = count;
    trainData1 = new double[num0*3];
    count = 0;
    for(int i=0;i<foregroundMask.rows;i++)
        for(int j=0;j<foregroundMask.cols;j++)
        {
            if(foregroundMask.at<cv::Vec3b>(i,j)[2] == 255 &&
                    foregroundMask.at<cv::Vec3b>(i,j)[1] == 0 &&
                    foregroundMask.at<cv::Vec3b>(i,j)[0] == 0)
            {
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[0];
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[1];
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[2];
            }
        }

    trainData2 = new double[num1 * 3];
    count = 0;
    for(int i=0;i<backgroundMask.rows;i++)
        for(int j=0;j<backgroundMask.cols;j++)
        {
            if(backgroundMask.at<cv::Vec3b>(i,j)[2] == 0 &&
                    backgroundMask.at<cv::Vec3b>(i,j)[1] == 0 &&
                    backgroundMask.at<cv::Vec3b>(i,j)[0] == 255)
            {
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[0];
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[1];
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[2];
            }
        }


    VlGMM *gmm;
    VlGMM *gmm2;
    double loglikelihood ;
    int dimension = 3;
    numClusters = 64;
    // create a new instance of a GMM object for float data
    gmm = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
    gmm2 = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
    // set the maximum number of EM iterations to 100
    vl_gmm_set_max_num_iterations (gmm, 100) ;
    vl_gmm_set_max_num_iterations (gmm2, 100) ;
    // set the initialization to random selection
    vl_gmm_set_initialization (gmm,VlGMMRand);
    vl_gmm_set_initialization (gmm2,VlGMMRand);
    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster (gmm, trainData1, num0);
    vl_gmm_cluster (gmm2, trainData2, num1);
    // get the means, covariances, and priors of the GMM
    means1 = (double*)vl_gmm_get_means(gmm);
    means2 = (double*)vl_gmm_get_means(gmm2);
    covariances1 = (double*)vl_gmm_get_covariances(gmm);
    covariances2 = (double*)vl_gmm_get_covariances(gmm2);
    priors1 = (double*)vl_gmm_get_priors(gmm);
    priors2 = (double*)vl_gmm_get_priors(gmm2);
    // get loglikelihood of the estimated GMM
//    loglikelihood = vl_gmm_get_loglikelihood(gmm) ;
    // get the soft assignments of the data points to each cluster
//    posteriors = (double*)vl_gmm_get_posteriors(gmm) ;

    printf("mask ok ... \n");


}

void GraphCut::test()
{
    cv::Vec3d tmp = image.at<cv::Vec3b>(499,999);
    double res = computeGMM(tmp[0],tmp[1],tmp[2]);
    printf("res. .. %le\n",res);
}

void GraphCut::erode(int kernelSize, int opTimes)
{
    this->kernelSize = kernelSize;

    this->opTimes = opTimes;

    cv::Mat kernel = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                         cv::Size( 2*kernelSize + 1, 2*kernelSize+1 ),
                                         cv::Point( kernelSize, kernelSize ) );
    mask.copyTo(backgroundMask);

    cv::threshold(backgroundMask,backgroundMask,128,255,cv::THRESH_BINARY_INV);

    for(int i=0;i<opTimes;i++)
        cv::erode(backgroundMask,backgroundMask,kernel);
//    cv::namedWindow("backgroundMask");
//    cv::imshow("backgroundMask",backgroundMask);
    printf("background mask....ok\n");
}

void GraphCut::dilate(int kernelSize, int opTimes)
{
    this->kernelSize = kernelSize;

    this->opTimes = opTimes;

    cv::Mat kernel = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                         cv::Size( 2*kernelSize + 1, 2*kernelSize+1 ),
                                         cv::Point( kernelSize, kernelSize ) );
    mask.copyTo(foregroundMask);

    cv::threshold(foregroundMask,foregroundMask,128,255,cv::THRESH_BINARY_INV);

    for(int i=0;i<opTimes;i++)
        cv::dilate(foregroundMask,foregroundMask,kernel);
//    cv::namedWindow("foregroundMask");
    //    cv::imshow("foregroundMask",foregroundMask);

    cv::threshold(foregroundMask,foregroundMask,128,255,cv::THRESH_BINARY_INV);
    cv::Mat tmp;
    image.copyTo(tmp);
    for(int i=0;i<image.rows;i++)
    {
        for(int j =0;j<image.cols;j++)
        {
            if(foregroundMask.at<uchar>(i,j) == 255)
            {
                tmp.at<cv::Vec3b>(i,j)[0] = 255;
                tmp.at<cv::Vec3b>(i,j)[1] = 255;
                tmp.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }

    printf("foreground mask....ok\n");
//        cv::namedWindow("foregroundMask");
//        cv::imshow("foregroundMask",tmp);

}

void GraphCut::setForegroundMask()
{
    int count = 0;
    int num0 = 0;

    printf("setForegroundMask ..1\n");

    num0  = 0;
    for(int i=0;i<foregroundMask.rows;i++)
        for(int j=0;j<foregroundMask.cols;j++)
            if(foregroundMask.at<uchar>(i,j) == 255)
                num0 ++;

    while(!num0)
    {
        kernelSize -= 3;
        opTimes -= 2;
        if(kernelSize <= 1 )
            kernelSize = 1;
        if(opTimes <= 1)
            opTimes = 1;
        dilate(kernelSize,opTimes);

        for(int i=0;i<foregroundMask.rows;i++)
            for(int j=0;j<foregroundMask.cols;j++)
                if(foregroundMask.at<uchar>(i,j) == 255)
                    num0 ++;

//        printf("foregroundMask num ... %d %d\n",num0,foregroundMask.rows*foregroundMask.cols);
    }

    printf("setForegroundMask ..2\n");
    trainData1 = new double[num0*3];
    for(int i=0;i<foregroundMask.rows;i++)
        for(int j=0;j<foregroundMask.cols;j++)
        {
            if(foregroundMask.at<uchar>(i,j) == 255)
            {
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[0];
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[1];
                trainData1[count++] = image.at<cv::Vec3b>(i,j)[2];
            }
        }
    printf("setForegroundMask ..3\n");
    VlGMM *gmm;
    int dimension = 3;

    gmm = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
    printf("setForegroundMask ..3.1\n");
    // set the maximum number of EM iterations to 100
    vl_gmm_set_max_num_iterations (gmm, 100) ;
    printf("setForegroundMask ..3.2\n");
    // set the initialization to random selection
    vl_gmm_set_initialization (gmm,VlGMMRand);
    printf("setForegroundMask ..3.3\n");
    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster (gmm, trainData1, num0);
    printf("num ... %d\n",num0);
    printf("setForegroundMask ...3.4\n");
    // get the means, covariances, and priors of the GMM
    means1 = (double*)vl_gmm_get_means(gmm);
    printf("setForegroundMask ..3.5\n");
    covariances1 = (double*)vl_gmm_get_covariances(gmm);
    printf("setForegroundMask ..3.6\n");
    priors1 = (double*)vl_gmm_get_priors(gmm);

    printf("foreground mask ok...\n");

//    delete gmm;
}

void GraphCut::setBackgroundMask()
{
    int count = 0;
    int num1 = 0;

    for(int i=0;i<backgroundMask.rows;i++)
        for(int j=0;j<backgroundMask.cols;j++)
            if(backgroundMask.at<uchar>(i,j) == 255)
                num1++;

    printf("set backgroundMask ..1\n");
    while(!num1){
//        num1 = 0;

        kernelSize -= 3;
        opTimes -= 2;
        if(kernelSize <= 1 )
            kernelSize = 1;
        if(opTimes <= 1)
            opTimes = 1;
        erode(kernelSize, opTimes);

        for(int i=0;i<backgroundMask.rows;i++)
            for(int j=0;j<backgroundMask.cols;j++)
                if(backgroundMask.at<uchar>(i,j) == 255)
                    num1++;
//        printf("backgroundMask ... %d\n",num1);
    }
    printf("set backgroundMask ..2\n");
    trainData2  = new double[num1 * 3];
    for(int i=0;i<backgroundMask.rows;i++)
        for(int j=0;j<backgroundMask.cols;j++)
        {
            if(backgroundMask.at<uchar>(i,j) == 255)
            {
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[0];
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[1];
                trainData2[count++] = image.at<cv::Vec3b>(i,j)[2];
            }
        }
    printf("set backgroundMask ..3\n");
    int dimension =3;
    VlGMM *gmm2;
    gmm2 = vl_gmm_new (VL_TYPE_DOUBLE, dimension, numClusters) ;
    // set the maximum number of EM iterations to 100
    vl_gmm_set_max_num_iterations (gmm2, 100) ;
    // set the initialization to random selection
    vl_gmm_set_initialization (gmm2,VlGMMRand);
    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster (gmm2, trainData2, num1);
    // get the means, covariances, and priors of the GMM
    means2 = (double*)vl_gmm_get_means(gmm2);
    covariances2 = (double*)vl_gmm_get_covariances(gmm2);
    priors2 = (double*)vl_gmm_get_priors(gmm2);

//    delete gmm2;
    printf("set backgroundMask ok\n");
}

int GraphCut::getVecdis(cv::Vec3b a, cv::Vec3b b)
{
    cv::Vec3b diff = a - b;
    int res = 0;
    for(int i=0;i<3;i++)
        res += (int)diff[i]*(int)diff[i];
    printf("res.... %d\n",res);
    return res;
}

double GraphCut::computeGMM(double r,double g,double b)
{      
    double res = 0.0;
    for(int i=0;i<numClusters;i++)
    {
        double dis = 0;
        dis = -((r - means1[i*3]) * (r - means1[i*3]) / covariances1[i*3] +
                (g - means1[i*3+1]) * (g - means1[i*3+1])  / covariances1[i*3+1] +
                (b - means1[i*3+2]) * (b - means1[i*3+2])  / covariances1[i*3+2])/2.0;
        res += priors1[i]*exp(dis) /
                sqrt(8.0 * PI * PI * PI * covariances1[i*3] * covariances1[i*3+1] * covariances1[i*3+2]);
    }
    return res;
}

double GraphCut::computeGMM2(double r, double g, double b)
{
    double res = 0.0;
    for(int i=0;i<numClusters;i++)
    {
        double dis = 0;
        dis = -((r - means2[i*3]) * (r - means2[i*3]) / covariances2[i*3] +
                (g - means2[i*3+1]) * (g - means2[i*3+1])  / covariances2[i*3+1] +
                (b - means2[i*3+2]) * (b - means2[i*3+2])  / covariances2[i*3+2])/2.0;
        res += priors2[i]*exp(dis) /
                sqrt(8.0 * PI * PI * PI * covariances2[i*3] * covariances2[i*3+1] * covariances2[i*3+2]);
    }
    return res;
}

void on_mouse(int event, int x, int y, int flags, void* param)
{
    static bool s_bMouseLButtonDown = false;
    static bool s_bMouseRButtonDown = false;
    static cv::Point s_cvPrePoint = cv::Point(-1, -1);
    static cv::Point s_cvCurPoint = cv::Point(-1, -1);

    switch (event)
    {
    case CV_EVENT_LBUTTONDOWN:
        s_bMouseLButtonDown = true;
        s_cvPrePoint = cv::Point(x,y);
        break;

    case CV_EVENT_RBUTTONDOWN:
        s_bMouseRButtonDown = true;
        s_cvPrePoint = cv::Point(x,y);
        break;

    case  CV_EVENT_LBUTTONUP:
        s_bMouseLButtonDown = false;
        break;
    case CV_EVENT_RBUTTONUP:
        s_bMouseRButtonDown = false;
        break;

    case CV_EVENT_MOUSEMOVE:
        if (s_bMouseLButtonDown)
        {
            s_cvCurPoint = cv::Point(x, y);
            cv::line(*(cv::Mat*)param, s_cvPrePoint, s_cvCurPoint, CV_RGB(255, 0, 0), 3,CV_AA,0);
//            cv::line(graphCutInstance->foregroundMask, s_cvPrePoint, s_cvCurPoint, 255,  3,CV_AA,0);
            graphCutInstance->foregroundMask = *(cv::Mat*)param;
            s_cvPrePoint = s_cvCurPoint;

            cv::imshow("pstrWindowsMouseDrawTitle", *(cv::Mat*)param);
        }
        if (s_bMouseRButtonDown)
        {
            s_cvCurPoint = cv::Point(x, y);
            cv::line(*(cv::Mat*)param, s_cvPrePoint, s_cvCurPoint, CV_RGB(0, 0, 255), 3,CV_AA,0);
//            cv::line(graphCutInstance->backgroundMask, s_cvPrePoint, s_cvCurPoint, 255,  3,CV_AA,0);
            graphCutInstance->backgroundMask = *(cv::Mat*)param;
            s_cvPrePoint = s_cvCurPoint;
            cv::imshow("pstrWindowsMouseDrawTitle", *(cv::Mat*)param);
        }
        break;
    }
}
