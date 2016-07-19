#include "fea.hh"
///
/// \brief Fea::viewpointSample
///     this function was created to determine camera parameters to generate some sample viewpoints
///     glm::lookAt function determines by camera position, camera towards and camera upwards direction
///
///     camera upwards direction was fixed to 0,1,0
///     what we should determine is camera position and camera towards.
///     to simplify this problem, we solve this problem in ballcoordinate.
///
///     As we get several frames from video sequence and get their model view matrix
///     by decompose model view matrix we can get their camera's position.
///     by this way, I take their average distance to object as the extimated distance.
///
///     by adjust the distance and the rotate angle to smaple some viewpoints.
///
/// \param fileInfo contains config file path
///
void Fea::viewpointSample(QString v_matrixPath, int sampleIndex, int numSamples, QString output, QString configPath)
{
    // prepare for parameters
    float distance = 0.f;
    float distanceStep = 0.f;
    int width = 0;
    int height = 0;
    vpSamplePrepare(v_matrixPath,distance,distanceStep,width,height);
    // generate Images ID
    int count  = 10000;


    // adjust center
    float xcenter = (render->p_xmin + render->p_xmax) / 2.0;
    float ycenter = (render->p_ymin + render->p_ymax) / 2.0;
    float zcenter = (render->p_zmin + render->p_zmax) / 2.0;
    float zcenterStep = (render->p_zmax - render->p_zmin) / 100;
    float xcenterStep = (render->p_xmax - render->p_zmin) / 100;
    // 图像bounding box上下边界和左右边界距图像边界的距离差与图像长或宽之比
    float accCenterWidthRate = 0.1;
    float accCenterHeightRate = 0.1;
    float caWidthCenterRate = 1.0;
    float caHeightCenterRate = 1.0;

    int bottom,up,left,right;
    roundingBox2D(up,bottom,left,right);
    std::cout << "bottom up left right "<< bottom << " " << up << " " << left << " " << right << std::endl;

    float acceptAreaRate = 0.2;
    float caAreaRate = (bottom - up) * (right - left) / (float)(width * height);
    // 初始时图像就是居中的，所以此时仅仅需要调整距离使得整个建筑物落入视口范围内
    float tmpDistance = distance;
    glm::mat4 v_model(1.0), v_view(1.0);
    glm::mat4 proj = m_projectionList[0];

    while(1)
    {
        image.release();
        mask.release();
        image2D.release();
        gray.release();
        // 整个建筑物都落入视口范围内
        if(up > 0.2 * height &&
                (height - bottom) > 0.2 * height &&
                left > 0.2 * width &&
                (width - right) > 0.2 * width)
        {
            break;
        }
        tmpDistance += distanceStep;
        v_view = glm::lookAt(glm::vec3(0,-tmpDistance,0),
                             glm::vec3(xcenter,ycenter,zcenter),
                             glm::vec3(0,0,1));
        render->setMVP(v_model,v_view,proj);
        // 渲染
        render->rendering(0);
        // 设置渲染之后的参数
        render->setParameters();
        setMat(render->p_img,render->p_width,render->p_height,width,height);
        roundingBox2D(up,bottom,left,right);
        std::cout << "bottom up left right "<< bottom << " " << up << " " << left << " " << right << std::endl;
        cv::Mat smallImage;
        cv::resize(image,smallImage,cv::Size(),0.05,0.05);
        cv::imshow("test",smallImage);
        cv::waitKey(0);
        smallImage.release();
    }

    while(1)
    {
        // 包围盒所占图像比例达到acceptRate
        // 图像居中比例小于accCenterRate
        if(caAreaRate > acceptAreaRate && caWidthCenterRate < accCenterWidthRate && caHeightCenterRate < accCenterHeightRate)
            break;
        // adjust boundbox size
        while(1)
        {
            if(caAreaRate > acceptAreaRate)
                break;
            image.release();
            mask.release();
            image2D.release();
            gray.release();
            tmpDistance -= distanceStep;
            v_view = glm::lookAt(glm::vec3(0,-tmpDistance,0),
                                 glm::vec3(xcenter,ycenter,zcenter),
                                 glm::vec3(0,0,1));
            render->setMVP(v_model,v_view,proj);
            // 渲染
            render->rendering(0);
            // 设置渲染之后的参数
            render->setParameters();
            setMat(render->p_img,render->p_width,render->p_height,width,height);
            roundingBox2D(up,bottom,left,right);
            caAreaRate = (bottom - up)*(right - left) / (float)(width * height);
            std::cout << "accRate " << caAreaRate << std::endl;
            std::cout << "bottom up left right "<< bottom << " " << up << " " << left << " " << right << std::endl;
            cv::Mat smallImage;
            cv::resize(image,smallImage,cv::Size(),0.05,0.05);
            cv::imshow("test",smallImage);
            cv::waitKey(0);
            smallImage.release();
        }
        // adjust direction
        while(1)
        {
            image.release();
            mask.release();
            image2D.release();
            gray.release();
            v_view = glm::lookAt(glm::vec3(0,-tmpDistance,0),
                                 glm::vec3(xcenter,ycenter,zcenter),
                                 glm::vec3(0,0,1));
            render->setMVP(v_model,v_view,proj);
            // 渲染
            render->rendering(0);
            // 设置渲染之后的参数
            render->setParameters();
            setMat(render->p_img,render->p_width,render->p_height,width,height);
            roundingBox2D(up,bottom,left,right);
            caWidthCenterRate = std::abs(left - width + right) / (float)width;
            caHeightCenterRate = std::abs(height - bottom - up) / (float)height;
            std::cout << "bottom up left right "<< bottom << " " << up << " " << left << " " << right << std::endl;
            std::cout << "accWidth Rate "<<caWidthCenterRate << " accHeight Rate " << caHeightCenterRate << std::endl;


            //        cv::imshow("test",image);
            //        cv::waitKey(0);

            if(caWidthCenterRate < accCenterWidthRate && caHeightCenterRate < accCenterHeightRate)
                break;
            if(caHeightCenterRate >= accCenterHeightRate)
            {
                if(up < (height - bottom))
                {
                    zcenter += zcenterStep;
                }
                else
                {
                    zcenter -= zcenterStep;
                }
            }
            if(caWidthCenterRate < accCenterWidthRate)
            {
                if(left < (width - right))
                {
                    xcenter -= xcenterStep;
                }
                else
                {
                    xcenter += xcenterStep;
                }
            }
        }
    }

    mask.release();
    image2D.release();
    gray.release();
    if(image.cols > 0 && image.rows > 0)
    cv::imwrite(configPath.append("/sample.jpg").toStdString(),image);

    const float X_LEN = 16.f;
    const float Z_LEN = 64.f;
    float angle_x = glm::pi<float>() / 90.0 / X_LEN;
    float angle_z = glm::pi<float>() / 2.0 / Z_LEN;

    // reset again
    v_view = glm::lookAt(glm::vec3(0,-tmpDistance,0),
                         glm::vec3(xcenter,ycenter,zcenter),
                         glm::vec3(0,0,1));

    std::ofstream fout(output.toStdString().c_str());
    glm::mat4 mv;

    // without visit different distance
    for(int i=0;i<X_LEN;i++)
    {
        for(int j=0;j<Z_LEN;j++)
        {
            // rotate with x axis
            float anglex = angle_x * i;
            // rotate with z axis
            float anglez = angle_z * j - glm::pi<float>() / 4.0;
            glm::mat4 rotateX = glm::rotate(glm::mat4(1.f),anglex,glm::vec3(1.0,0.0,0.0));
            glm::mat4 rotateZ = glm::rotate(glm::mat4(1.f),anglez,glm::vec3(0.0,0.0,1.0));
            mv = v_view *rotateX * rotateZ;
            fout << "img";
            fout.width(8);
            fout.fill('0');
            fout << count++ << ".jpg" << std::endl;

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

    std::cout << "export done" << std::endl;



}

void Fea::vpSamplePrepare(QString v_matrixPath, float &distance, float &distanceStep,int &width,int &height)
{
    // read in matrixFile
    setMvpPara(v_matrixPath);
    std::vector<glm::vec3> v_eye;
    std::vector<glm::vec3> v_center;
    std::vector<glm::vec3> v_up;
    deComposeMV(v_eye,v_center,v_up);

    glm::vec3 meanCenter(0.f,0.f,0.f);
    glm::vec3 meanEye(0.f,0.f,0.f);
    for(int i=0;i<v_center.size();i++)
    {
        meanCenter.x += v_center[i].x;
        meanCenter.y += v_center[i].y;
        meanCenter.z += v_center[i].z;

        meanEye.x += v_eye[i].x;
        meanEye.y += v_eye[i].y;
        meanEye.z += v_eye[i].z;
    }

    meanCenter.x /= v_center.size();
    meanCenter.y /= v_center.size();
    meanCenter.z /= v_center.size();

    meanEye.x /= v_center.size();
    meanEye.y /= v_center.size();
    meanEye.z /= v_center.size();

    // 参考距离，在这个距离的基础上进行变换
    distance = glm::length(meanEye);
    std::cout << "mean distance "<< distance << std::endl;
    // 距离变化步长
    distanceStep = distance * 0.1;

    glm::mat4 v_model(1.0), v_view(1.0);
    v_view = glm::lookAt(glm::vec3(0,-distance,0),
                         glm::vec3(0,0,0),
                         glm::vec3(0,0,1));
    // 设置渲染参数MVP matrix
    glm::mat4 proj = m_projectionList[0];
    render->setMVP(v_model,v_view,proj);
    // 渲染
    render->rendering(0);
    // 设置渲染之后的参数
    render->setParameters();
    width = 0;
    height  = 0;
    // 参考图像，用来确定图像的长宽
    image2D = cv::imread(fileName.at(0).toStdString().c_str());
    width = image2D.cols;
    height = image2D.rows;

    //    render->storeImage(path,QString("img").append(QString::number(count)).append(".jpg"),width,height);
    // 得到渲染用当前参数渲染之后的图像
    setMat(render->p_img,render->p_width,render->p_height,width,height);
    // setMask 是为了让setProjectArea的时候可以保存下来mask图像
    //    setMask();
    //    setProjectArea();
    //    setOutlierCount();
}

void Fea::vpSampleWholeArchitecture()
{

}

void Fea::vpSampleArchitectureSize()
{

}

void Fea::vpSampleArchitectureCenter()
{

}
