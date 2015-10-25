#include "render.hh"
#include "meshglhelper.hh"
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QCoreApplication>
#include <math.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include "shader.hh"
#include "trackball.hh"
#include "gausscurvature.hh"
#include "meancurvature.hh"
#include <opencv.hpp>
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
        delete p_img;
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
    std::cout<<"********all initial complete"<<std::endl;
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
    qDebug()<<"FramebufferName fbo...init"<<endl;

    m_programID = LoadShaders("simpleShader.vert","simpleShader.frag");
    GLuint vertexNormal_modelspaceID = glGetAttribLocation(m_programID, "vertexNormal_modelspace");
    GLuint vertexPosition_modelspaceID = glGetAttribLocation(m_programID,"vertexPosition_modelspace");
    m_helper.fbo_init(vertexPosition_modelspaceID,vertexNormal_modelspaceID);
}

void Render::resizeGL(int width, int height)
{
    std::cout << "resize " << width << " " << height << std::endl;
    //glViewport(0,0,800,800);
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
    qDebug()<<" "<<count<<endl;
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

    qDebug()<<"show fbo info...ok"<<endl;

    cv::Mat image = cv::Mat(viewport[3],viewport[2],CV_8UC4,img);
    qDebug()<<"show fbo info...ok"<<endl;
    cv::namedWindow("test");
    imshow("test",image);



    glBindFramebuffer(GL_FRAMEBUFFER,0);


    doneCurrent();

}
// fileName is absolute name
// fileName  = path + name
void Render::storeImage(QString path,QString fileName)
{
    makeCurrent();

    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    std::cout << viewport[0] << " " << viewport[1] << " " << viewport[2] <<  " " << viewport[3] << std::endl;

    GLfloat *img0 = new GLfloat[(viewport[2]-viewport[0])*(viewport[3]-viewport[1])];
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0,0,viewport[2],viewport[3],GL_DEPTH_COMPONENT,GL_FLOAT,img0);

    cv::Mat depthImgFliped = cv::Mat(viewport[3],viewport[2],CV_32FC1,img0);
    cv::Mat depthImg;
    cv::flip(depthImgFliped,depthImg,0);
    depthImg.convertTo(depthImg,CV_8UC1,255,0);

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
    cv::cvtColor(rgbImg,rgbImg,CV_RGBA2BGR);
    rgbImg.convertTo(rgbImg,CV_8UC3);

    // fileName  = path + name
//    int len = path.length();
    QString outputDepthFile = fileName;
    outputDepthFile.append(".jpg");
    QString outputDepth = path;
//    outputDepthFile.remove(0,len);
    outputDepth.append("depth/");
    outputDepth.append(outputDepthFile);
    std::cout<< outputDepth.toStdString() << std::endl;
    std::cout<<depthImg.type()<<std::endl;
    // imwrite has a bug maybe need recompile
    // ref http://stackoverflow.com/questions/6923296/opencv-imwrite-2-2-causes-exception-with-message-opencv-error-unspecified-erro
//      cv::imwrite(outputDepth.toStdString(),depthImg);
    cvSaveImage(outputDepth.toStdString().c_str(),&(IplImage(depthImg)));

    QString outputRgbFile = fileName;
    outputRgbFile.append(".jpg");
    QString outputRgb = path;
//    outputRgbFile.remove(0,len);
    outputRgb.append("rgb/");
    outputRgb.append(outputRgbFile);
    std::cout<< outputRgb.toStdString() << std::endl;
//      cv::imwrite(outputRgb.toStdString(),rgbImg);
    cvSaveImage(outputRgb.toStdString().c_str(),&(IplImage(rgbImg)));


    delete []img0;
    depthImgFliped.release();
    depthImg.release();
    delete []img;
    rgbaImgFliped.release();
    rgbImg.release();
}

void Render::setParameters()
{
    std::vector<GLuint> indices;
//    p_vertices.clear();
//    p_isVertexVisible.clear();
//    p_VisibleFaces.clear();
//    p_verticesMvp.clear();
    std::vector<GLfloat>().swap(p_vertices);
    std::vector<bool>().swap(p_isVertexVisible);
    std::vector<GLuint>().swap(p_VisibleFaces);
    std::vector<GLfloat>().swap(p_verticesMvp);

    if(p_img)
    {
        delete p_img;
        p_img = NULL;
    }

    m_helper.getVerticesAndFaces_AddedByZwz(p_vertices,indices);
    makeCurrent();

    glBindFramebuffer(GL_FRAMEBUFFER,frameBufferId);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    glm::mat4 modelViewMatrix = getModelViewMatrix();
    glm::mat4 mvp = m_proj * modelViewMatrix;


    int visibleVertexCount = 0;
    for(int i=0;i<p_vertices.size();i+=3)
    {
        glm::vec4 position = mvp * glm::vec4(p_vertices[i],p_vertices[i+1],p_vertices[i+2],1.0);
        position = position / position.w;

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

        // 在3*3邻域内找相似的深度值
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++) {
                GLfloat winZ;
                glReadBuffer(GL_BACK);
                glReadPixels(ax+i, ay+j,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&winZ);

                // 它们的z-buffer值相差不大，表示这是一个可见点
                if (abs(winZ - finalZ) < 0.00015) {
                    isVisible = true;
                    break;
                }
            }
        p_isVertexVisible.push_back(isVisible);
        visibleVertexCount += isVisible ? 1 : 0;
    }

    // 筛选出可见面
    // 所谓可见面，就是指该面上其中一个顶点可见
    p_VisibleFaces.clear();
    for (int i = 0; i < indices.size(); i += 3)
        if (p_isVertexVisible[indices[i]]
                || p_isVertexVisible[indices[i+1]]
                || p_isVertexVisible[indices[i+2]]) {
            p_VisibleFaces.push_back(indices[i]);
            p_VisibleFaces.push_back(indices[i+1]);
            p_VisibleFaces.push_back(indices[i+2]);
        }
//    GLuint vertexPosition_modelspaceID = glGetAttribLocation(m_programID, "vertexPosition_modelspace");
//    m_helper.replace_init(vertices, VisibleFaces, vertexPosition_modelspaceID);
    p_width = viewport[3]-viewport[1];
    p_height = viewport[2]-viewport[0];
    p_img = new float[p_width*p_height];
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,p_height,p_width,GL_DEPTH_COMPONENT,GL_FLOAT,p_img);

    p_model = m_model;

    glBindFramebuffer(GL_FRAMEBUFFER,0);

    doneCurrent();

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

