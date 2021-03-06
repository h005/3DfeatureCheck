﻿#include "render.hh"
#include "meshglhelper.hh"
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QCoreApplication>
#include <math.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
//#include "shader.hh"
#include "trackball.hh"
#include "gausscurvature.hh"
#include "meancurvature.hh"
#include <assert.h>

Render::Render(MyMesh &in_mesh,
               QString fileName,
               QWidget *parent)
    : QOpenGLWidget(parent),
      m_mesh(in_mesh),
      m_helper(in_mesh),
      fileName(fileName)
{
    m_transparent = QCoreApplication::arguments().contains(QStringLiteral("--transparent"));
    if (m_transparent)
        setAttribute(Qt::WA_TranslucentBackground);

    // read model view and projection Matrix from .mvp file

    QString tmpPath = fileName;
    //    tmpPath.append(".mvp");

    FILE *fp;
    fp = freopen(tmpPath.toStdString().c_str(),"r",stdin);

    if(fp)
    {
        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                scanf("%f",&m_model[i][j]);

        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                scanf("%f",&m_view[i][j]);

        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                scanf("%f",&m_proj[i][j]);
    }
    else
    {
        m_model = glm::mat4();
        m_view = glm::mat4();
        m_proj = glm::mat4();
    }

    fclose(stdin);
    p_img = NULL;
    frameBufferId = 0;
    depthRenderBuffer = 0;
    colorRenderBuffer = 0;
}

Render::Render(MyMesh &in_mesh,
               glm::mat4 &model,
               glm::mat4 &view,
               glm::mat4 &projection,
               QWidget *parent)
    : QOpenGLWidget(parent),
      m_mesh(in_mesh),
      m_helper(in_mesh),
      m_model(model),
      m_view(view),
      m_proj(projection)
{
    m_transparent = QCoreApplication::arguments().contains(QStringLiteral("--transparent"));
    if (m_transparent)
        setAttribute(Qt::WA_TranslucentBackground);

    p_img = NULL;
    frameBufferId = 0;
    depthRenderBuffer = 0;
    colorRenderBuffer = 0;
}

void Render::setMVP(glm::mat4 &model, glm::mat4 &view, glm::mat4 &proj)
{
    m_model = model;
    m_view = view;
    m_proj = proj;
}


void Render::setMeshSaliencyPara(ExternalImporter<MyMesh> *exImporter)
{
    exImporter->setMeshVector(p_vecMesh,p_indiceArray);
}

Render::~Render()
{
    if(p_img)
        delete []p_img;
    p_vertices.clear();
    p_isVertexVisible.clear();
    p_VisibleFaces.clear();
    p_verticesMvp.clear();
}

void Render::cleanup()
{
    makeCurrent();
    if(m_programID)
    {
        glDeleteProgram(m_programID);
    }
    if(frameBufferId)
        glDeleteRenderbuffers(1,&frameBufferId);
    if(depthRenderBuffer)
        glDeleteRenderbuffers(1,&depthRenderBuffer);
    if(colorRenderBuffer)
        glDeleteRenderbuffers(1,&colorRenderBuffer);
    m_helper.cleanup();
    doneCurrent();
}

void Render::initializeGL()
{
    std::cout<<"initialGL"<<std::endl;
    //read file
    //set m_camera
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    assert(err == GLEW_OK);

    connect(context(),&QOpenGLContext::aboutToBeDestroyed,this,&Render::cleanup);
    initializeOpenGLFunctions();

    glClearColor( 0.368, 0.368, 0.733, 1);
    initial();
}



void Render::paintGL()
{
}

void Render::initial()
{
    if(frameBufferId == 0)
        glGenFramebuffers(1, &frameBufferId);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferId);
    if(depthRenderBuffer == 0)
        glGenRenderbuffers(1, &depthRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 800);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);

    if(colorRenderBuffer == 0);
        glGenRenderbuffers(1, &colorRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, colorRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, 800,800);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorRenderBuffer);

    glBindFramebuffer(GL_FRAMEBUFFER,0);
    std::cout << "loadShaders before" << std::endl;
    m_programID = LoadShaders("shader/simpleShader.vert","shader/simpleShader.frag");
    std::cout << "loadShaders done" << std::endl;
    GLuint vertexNormal_modelspaceID = glGetAttribLocation(m_programID, "vertexNormal_modelspace");
    GLuint vertexPosition_modelspaceID = glGetAttribLocation(m_programID,"vertexPosition_modelspace");
    m_helper.fbo_init(vertexPosition_modelspaceID,vertexNormal_modelspaceID);
}

GLuint Render::LoadShaders(const char *vertex_file_path, const char *fragment_file_path)
{

    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open()){
        std::string Line = "";
        while(getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }else{
        printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
        fflush(stdout);
        assert(0);
        return 0;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }


    GLint Result = GL_FALSE;
    int InfoLogLength;



    // Compile Vertex Shader
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);
    std::cout << ".............shader.............." << std::endl;
    std::cout << VertexSourcePointer << std::endl;
    std::cout << ".............shader.............." << std::endl;

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( !Result && InfoLogLength > 0 ){
        std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }



    // Compile Fragment Shader
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( !Result && InfoLogLength > 0 ){
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }



    // Link the program
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( !Result && InfoLogLength > 0 ){
        std::vector<char> ProgramErrorMessage(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

void Render::resizeGL(int width, int height)
{
    std::cout << "resize " << width << " " << height << std::endl;
}

QSize Render::sizeHint() const
{
    return QSize(800,800);
}

QSize Render::minimumSizeHint() const
{
    return QSize(800,800);
}

bool Render::rendering(int count)
{
    // makeCurrent() 在paintGL函数中会自动调用，所以要自己加上，不然会有bug
    makeCurrent();
    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_FLAT);
    glShadeModel(GL_FLAT);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    glm::mat4 modelViewMatrix = getModelViewMatrix();

    glm::mat4 normalMatrix = glm::transpose(glm::inverse(modelViewMatrix));
    glm::vec4 lightPos0 = glm::inverse(m_view) * glm::vec4(4,4,4,1);
    glm::vec3 lightPos = glm::vec3(lightPos0.x,lightPos0.y,lightPos0.z);

    glm::mat4 MVP = m_proj * modelViewMatrix;

    glUseProgram(m_programID);

    GLuint mvpID = glGetUniformLocation(m_programID,"MVP");
    GLuint mID = glGetUniformLocation(m_programID,"M");
    GLuint vID = glGetUniformLocation(m_programID,"V");
    GLuint nID = glGetUniformLocation(m_programID,"normalMatrix");
    GLuint lightID = glGetUniformLocation(m_programID,"LightPosition_worldspace");

    glUniformMatrix4fv(mvpID, 1, GL_FALSE, glm::value_ptr(MVP));
    glUniformMatrix4fv(mID, 1, GL_FALSE, glm::value_ptr(m_model));
    glUniformMatrix4fv(vID, 1, GL_FALSE, glm::value_ptr(m_view));
    glUniformMatrix4fv(nID, 1, GL_FALSE, glm::value_ptr(normalMatrix));
    glUniform3f(lightID, lightPos.x, lightPos.y, lightPos.z);

    m_helper.draw();
    doneCurrent();
    return true;
}


void Render::showImage()
{
    makeCurrent();

    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    GLfloat *img0 = new GLfloat[(viewport[2]-viewport[0])*(viewport[3]-viewport[1])];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,0,viewport[2],viewport[3],GL_DEPTH_COMPONENT,GL_FLOAT,img0);

    cv::Mat image0 = cv::Mat(viewport[3],viewport[2],CV_32FC1,img0);
    cv::namedWindow("test0");
    imshow("test0",image0);


    GLubyte *img =
            new GLubyte[(viewport[2]-viewport[0])
            *(viewport[3]-viewport[1])*4];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,
            0,
            viewport[2],
            viewport[3],
            GL_BGRA,
            GL_UNSIGNED_BYTE,
            img);


    cv::Mat image = cv::Mat(viewport[3],viewport[2],CV_8UC4,img);

    cv::namedWindow("test");
    imshow("test",image);

    glBindFramebuffer(GL_FRAMEBUFFER,0);


    doneCurrent();

}
    // fileName is absolute name
    // fileName  = path + name
void Render::storeImage(QString path,QString fileName0,int width,int height)
{
    int pos = fileName0.lastIndexOf('/');
    QString fileName = fileName0.remove(0,pos+1);

    makeCurrent();

    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    GLfloat *img0 = new GLfloat[(viewport[2]-viewport[0])*(viewport[3]-viewport[1])];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,0,viewport[2],viewport[3],GL_DEPTH_COMPONENT,GL_FLOAT,img0);

    float min = 1.0;
    for(int i=0;i<viewport[2]*viewport[3];i++)
        min = min < img0[i] ? min : img0[i];

    cv::Mat depthImgFliped = cv::Mat(viewport[3],viewport[2],CV_32FC1,img0);
    cv::Mat depthImg;
    cv::flip(depthImgFliped,depthImg,0);
    cv::resize(depthImg,depthImg,cv::Size(width,height));
    depthImg.convertTo(depthImg,CV_8UC1,255.0 / (1.0 - min),255.0 * min / (min - 1.0));

    GLubyte *img =
            new GLubyte[(viewport[2] - viewport[0])
            *(viewport[3] - viewport[1])*4];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,
            0,
            viewport[2],
            viewport[3],
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            img);
    cv::Mat rgbaImgFliped = cv::Mat(viewport[3],viewport[2],CV_8UC4,img);
    cv::Mat rgbImg;
    cv::flip(rgbaImgFliped,rgbImg,0);
    cv::resize(rgbImg,rgbImg,cv::Size(width,height));
    cv::cvtColor(rgbImg,rgbImg,CV_RGBA2BGR);
    rgbImg.convertTo(rgbImg,CV_8UC3);

    // fileName  = path + name
    // int len = path.length();
    QString outputDepthFile = fileName;
    outputDepthFile.append(".jpg");
    QString outputDepth = path;

    outputDepth.append("depth/");
    outputDepth.append(outputDepthFile);
    // imwrite has a bug maybe need recompile
    // ref http://stackoverflow.com/questions/6923296/opencv-imwrite-2-2-causes-exception-with-message-opencv-error-unspecified-erro
    //      cv::imwrite(outputDepth.toStdString(),depthImg);
    IplImage *saveImage = new IplImage(depthImg);
    cvSaveImage(outputDepth.toStdString().c_str(),saveImage);
    delete saveImage;
    QString outputRgbFile = fileName;
    outputRgbFile.append(".jpg");
    QString outputRgb = path;
    //    outputRgbFile.remove(0,len);
    outputRgb.append("rgb/");
    outputRgb.append(outputRgbFile);
    //      cv::imwrite(outputRgb.toStdString(),rgbImg);
    saveImage = new IplImage(rgbImg);
    cvSaveImage(outputRgb.toStdString().c_str(),saveImage);


    delete []img0;
    depthImgFliped.release();
    depthImg.release();
    delete []img;
    rgbaImgFliped.release();
    rgbImg.release();
}

double Render::getArea(std::vector<GLuint> &indices,int p)
{
    double area = 0.0;

    glm::vec3 pa = glm::vec3(p_vertices[indices[p]*3],p_vertices[indices[p]*3+1],p_vertices[indices[p]*3+2]);
    glm::vec3 pb = glm::vec3(p_vertices[indices[p+1]*3],p_vertices[indices[p+1]*3+1],p_vertices[indices[p+1]*3+2]);
    glm::vec3 pc = glm::vec3(p_vertices[indices[p+2]*3],p_vertices[indices[p+2]*3+1],p_vertices[indices[p+2]*3+2]);

    glm::vec3 BC = pc - pb;
    double a = glm::l2Norm(BC);

    glm::vec3 CA = pa - pc;
    double b = glm::l2Norm(CA);

    glm::vec3 BA = pa - pb;
    double c = glm::l2Norm(BA);

    if(glm::l2Norm(glm::cross(BA,BC)) < 1e-2)
        return 0;
    double s = (a + b + c) / 2.0;
//    qDebug()<<"pca s "<<s<<endl;
//    qDebug()<<"pca ss "<<s*(s-a)*(s-b)*(s-c)<<endl;
    // pca ss  -1.42503e-09
    // I am drunk!
    double tmpres = s*(s-a)*(s-b)*(s-c);
    if(abs(tmpres) < 1e-5)
        area = sqrt(0);
    else
        area = sqrt(tmpres);
//    qDebug()<<"pca area ... "<<area;

    return area;
}

void Render::setParameters()
{
    std::vector<GLuint> indices;
    p_vertices.clear();
    p_isVertexVisible.clear();
    p_VisibleFaces.clear();
    p_verticesMvp.clear();
    p_outsidePointsNum = 0;

    if(p_img)
    {
        delete []p_img;
        p_img = NULL;
    }

    m_helper.getVerticesAndFaces_AddedByZwz(p_vertices,indices);

    makeCurrent();

    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    glm::mat4 modelViewMatrix = getModelViewMatrix();
    glm::mat4 mvp = m_proj * modelViewMatrix;


    p_height = viewport[3]-viewport[1];
    p_width = viewport[2]-viewport[0];
    p_img = new GLfloat[p_width*p_height];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,0,p_width,p_height,GL_DEPTH_COMPONENT,GL_FLOAT,p_img);

    int visibleVertexCount = 0;

    GLfloat xmin,xmax,ymin,ymax,zmin,zmax;

    xmin = xmax = p_vertices[0];
    ymin = ymax = p_vertices[1];
    zmin = zmax = p_vertices[2];

    for(int i=0;i<p_vertices.size();i+=3)
    {
        glm::vec4 position = mvp * glm::vec4(p_vertices[i],p_vertices[i+1],p_vertices[i+2],1.0);
        position = position / position.w;


        xmin = xmin < p_vertices[i] ? xmin : p_vertices[i];
        xmax = xmax > p_vertices[i] ? xmax : p_vertices[i];
        ymin = ymin < p_vertices[i+1] ? ymin : p_vertices[i+1];
        ymax = ymax > p_vertices[i+1] ? ymax : p_vertices[i+1];
        zmin = zmin < p_vertices[i+2] ? zmin : p_vertices[i+2];
        zmax = zmax > p_vertices[i+2] ? zmax : p_vertices[i+2];
        // 看来读到的z-buffer并不是position.z，而是将position.z变换到[0, 1]之间
        // ref http://gamedev.stackexchange.com/a/18858
        GLfloat finalZ = position.z * 0.5 + 0.5;


        // 假设所有点都在裁剪平面内，1.off符合
        // TODO: position.x, position.y的边界检查
        GLfloat ax = (position.x + 1) / 2 * viewport[2];
        GLfloat ay = (position.y + 1) / 2 * viewport[3];
        p_verticesMvp.push_back(ax);
        p_verticesMvp.push_back(ay);
        p_verticesMvp.push_back(finalZ);
        bool isVisible = false;

        bool pointOutsideScreen = true;
        // 在3*3邻域内找相似的深度值
        for (int i = -1; i <= 1; i++){

            for (int j = -1; j <= 1; j++) {

                int x = (int)ax + i, y = (int)ay + j;
                if (x >= 0 && x < p_width && y >= 0 && y < p_height) {
                    pointOutsideScreen = false;
                    GLfloat winZ = p_img[y * p_height + x];

                    // 它们的z-buffer值相差不大，表示这是一个可见点
                    GLfloat zAbs = winZ - finalZ > 0 ? winZ - finalZ : finalZ - winZ;
//                    std::cout << " z buffer value " << zAbs << std::endl;
                    if (zAbs < 0.015) {
                        isVisible = true;
                        break;
                    }
                }

            }
            if(isVisible)
                break;
        }
        if (pointOutsideScreen) {
            // 渲染出的点在可视区域外
            p_outsidePointsNum++;
        }
        p_isVertexVisible.push_back(isVisible);
        visibleVertexCount += isVisible ? 1 : 0;
    }

    p_xmin = xmin;
    p_xmax = xmax;
    p_ymin = ymin;
    p_ymax = ymax;
    p_zmin = zmin;
    p_zmax = zmax;

    p_model_x = glm::vec4(xmax - xmin,0,0,0);
    p_model_y = glm::vec4(0,ymax - ymin,0,0);
    p_model_z = glm::vec4(0,0,zmax - zmin,0);
    p_model_x = modelViewMatrix * p_model_x;
    p_model_y = modelViewMatrix * p_model_y;
    p_model_z = modelViewMatrix * p_model_z;
    p_model_x[3] = 0;
    p_model_y[3] = 0;
    p_model_z[3] = 0;

    // 筛选出可见面
    // 所谓可见面，就是指该面上其中一个顶点可见
    p_VisibleFaces.clear();
    for (int i = 0; i < indices.size(); i += 3)
        if (p_isVertexVisible[indices[i]]
                && p_isVertexVisible[indices[i+1]]
                && p_isVertexVisible[indices[i+2]]) {
            p_VisibleFaces.push_back(indices[i]);
            p_VisibleFaces.push_back(indices[i+1]);
            p_VisibleFaces.push_back(indices[i+2]);
        }

    p_model = m_model;

    glBindFramebuffer(GL_FRAMEBUFFER,0);

    doneCurrent();

}

void Render::setVerticeNormals()
{
    m_helper.getVerticeNormals(p_verticeNormals);
}

void Render::setAreaAllFaces()
{
    std::vector<GLuint> indices;
    p_vertices.clear();

    m_helper.getVerticesAndFaces_AddedByZwz(p_vertices,indices);
    areaAllFaces = 0.0;

    for(int i=0; i< indices.size()-2;i++)
    {
        CvPoint3D64f p1 = cvPoint3D64f(p_vertices[3*indices[i]],p_vertices[3*indices[i]+1],p_vertices[3*indices[i]+2]);
        CvPoint3D64f p2 = cvPoint3D64f(p_vertices[3*indices[i+1]],p_vertices[3*indices[i+1]+1],p_vertices[3*indices[i+1]+2]);
        CvPoint3D64f p3 = cvPoint3D64f(p_vertices[3*indices[i+2]],p_vertices[3*indices[i+2]+1],p_vertices[3*indices[i+2]+2]);
        areaAllFaces += getArea3D(&p1,&p2,&p3);
    }
}

double Render::getArea3D(CvPoint3D64f *a, CvPoint3D64f *b, CvPoint3D64f *c)
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

void Render::clear()
{
    std::vector<GLfloat>().swap(p_vertices);
    std::vector<bool>().swap(p_isVertexVisible);
    std::vector<GLuint>().swap(p_VisibleFaces);
    std::vector<GLfloat>().swap(p_verticesMvp);
    delete []p_img;
}

glm::mat4 Render::getModelViewMatrix()
{
    return m_view*m_model;
}

glm::mat4 Render::getModelMatrix()
{
    return m_model;
}

glm::mat3 Render::getVecMatrix(glm::vec3 &v1)
{
    glm::mat3 res;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            res[j][i] = v1[i]*v1[j];
    return res;
}

