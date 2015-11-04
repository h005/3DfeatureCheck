#-------------------------------------------------
#
# Project created by QtCreator 2015-10-19T02:18:57
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG   += console

TARGET = 3DfeatureCheck
TEMPLATE = app

OTHER_FILES += shader/*.vert shader/*.frag

SOURCES += main.cpp\
        mainwindow.cpp \
    fea.cc \
    render.cc \
    shader.cc \
    trackball.cc \
    ufface.cpp

HEADERS  += mainwindow.h \
    abstractfeature.hh \
    colormap.hh \
    common.hh \
    curvature.hh \
    externalimporter.hh \
    fea.hh \
    gausscurvature.hh \
    meancurvature.hh \
    meshglhelper.hh \
    render.hh \
    shader.hh \
    trackball.hh \
    ufface.h \
    predefine.h

FORMS    += mainwindow.ui

DEFINES += _USE_MATH_DEFINES GLM_FORCE_RADIANS

win32 {
    DP_TOOLS_DIR = $$(DP_TOOLS_DIR)

    # OpenMesh
    INCLUDEPATH += $$DP_TOOLS_DIR/openmesh/include
    LIBS += -L$$DP_TOOLS_DIR/openmesh/lib/vs2013 -lOpenMeshCored

    # assimp
    LIBS += -L$$DP_TOOLS_DIR/assimp-3.1.1-win-binaries/build/code/Release -lassimp
    INCLUDEPATH += $$DP_TOOLS_DIR/assimp-3.1.1-win-binaries/include

    # glew
    INCLUDEPATH += $$DP_TOOLS_DIR/glew/include
    LIBS += -L$$DP_TOOLS_DIR/glew/lib/Release/Win32 -lglew32

    INCLUDEPATH += $$DP_TOOLS_DIR\opencv\build\include\opencv\
                    $$DP_TOOLS_DIR\opencv\build\include\opencv2\
                    $$DP_TOOLS_DIR\opencv\build\include

    LIBS += -L$$DP_TOOLS_DIR\opencv\build\x86\vc12\lib \
            -lopencv_calib3d249d \
            -lopencv_contrib249d \
            -lopencv_core249d \
            -lopencv_features2d249d \
            -lopencv_flann249d \
            -lopencv_gpu249d \
            -lopencv_highgui249d \
            -lopencv_imgproc249d \
            -lopencv_legacy249d \
            -lopencv_ml249d \
            -lopencv_nonfree249d \
            -lopencv_objdetect249d \
            -lopencv_ocl249d \
            -lopencv_photo249d \
            -lopencv_stitching249d \
            -lopencv_superres249d \
            -lopencv_ts249d \
            -lopencv_video249d \
            -lopencv_videostab249d
}

# glm
INCLUDEPATH += $$DP_TOOLS_DIR/glm

defineTest(copyToDestdir) {
    files = $$1

    for(FILE, files) {
        DDIR = $$OUT_PWD

        # Replace slashes in paths with backslashes for Windows
        win32:FILE ~= s,/,\\,g
        win32:DDIR ~= s,/,\\,g

        QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$FILE) $$quote($$DDIR) $$escape_expand(\\n\\t)
    }

    export(QMAKE_POST_LINK)
}
copyToDestdir($$_PRO_FILE_PWD_/shader)

