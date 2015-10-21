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
    ufface.h

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
}

# glm
INCLUDEPATH += D:/tools/glm

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

#opencv
INCLUDEPATH += D:\tools\opencv\build\include\opencv\
                D:\tools\opencv\build\include\opencv2\
                D:\tools\opencv\build\include

LIBS += D:/tools/opencv/build/x86/vc12/lib/*.lib

