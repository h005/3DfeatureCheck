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
    ufface.cpp \
    reverseface.cpp \
    spheregenerator.cpp \
    vpsample.cpp

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
    predefine.h \
    reverseface.h \
    spheregenerator.h

FORMS    += mainwindow.ui

DEFINES += _USE_MATH_DEFINES GLM_FORCE_RADIANS

#win32 {
#    DP_TOOLS_DIR = $$(DP_TOOLS_DIR)

#    # OpenMesh
#    INCLUDEPATH += $$DP_TOOLS_DIR/openmesh/include
#    LIBS += -L$$DP_TOOLS_DIR/openmesh/lib/vs2013 -lOpenMeshCored

#    # assimp
#    LIBS += -L$$DP_TOOLS_DIR/assimp-3.1.1-win-binaries/build/code/Release -lassimp
#    INCLUDEPATH += $$DP_TOOLS_DIR/assimp-3.1.1-win-binaries/include

#    # glew
#    INCLUDEPATH += $$DP_TOOLS_DIR/glew/include
#    LIBS += -L$$DP_TOOLS_DIR/glew/lib/Release/Win32 -lglew32

#    INCLUDEPATH += $$DP_TOOLS_DIR\opencv\build\include\opencv\
#                    $$DP_TOOLS_DIR\opencv\build\include\opencv2\
#                    $$DP_TOOLS_DIR\opencv\build\include

#    LIBS += -L$$DP_TOOLS_DIR\opencv\build\x86\vc12\lib \
#            -lopencv_calib3d249d \
#            -lopencv_contrib249d \
#            -lopencv_core249d \
#            -lopencv_features2d249d \
#            -lopencv_flann249d \
#            -lopencv_gpu249d \
#            -lopencv_highgui249d \
#            -lopencv_imgproc249d \
#            -lopencv_legacy249d \
#            -lopencv_ml249d \
#            -lopencv_nonfree249d \
#            -lopencv_objdetect249d \
#            -lopencv_ocl249d \
#            -lopencv_photo249d \
#            -lopencv_stitching249d \
#            -lopencv_superres249d \
#            -lopencv_ts249d \
#            -lopencv_video249d \
#            -lopencv_videostab249d
#}

## glm
#INCLUDEPATH += $$DP_TOOLS_DIR/glm

#defineTest(copyToDestdir) {
#    files = $$1

#    for(FILE, files) {
#        DDIR = $$OUT_PWD

#        # Replace slashes in paths with backslashes for Windows
#        win32:FILE ~= s,/,\\,g
#        win32:DDIR ~= s,/,\\,g

#        QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$FILE) $$quote($$DDIR) $$escape_expand(\\n\\t)
#    }

#    export(QMAKE_POST_LINK)
#}
#copyToDestdir($$_PRO_FILE_PWD_/shader)

# OpenMesh
INCLUDEPATH += /usr/local/include
LIBS += -lOpenMeshCore

# assimp
INCLUDEPATH += /usr/local/include/assimp
LIBS += -L/usr/local/lib/ -lassimp

# glew
# INCLUDEPATH +=
LIBS += -lGLEW -lGLU -lGL

# glm
INCLUDEPATH += usr/include/glm

#opencv
INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \

LIBS += /usr/local/lib/libopencv_aruco.so.3.1 \
        /usr/local/lib/libopencv_bgsegm.so.3.1 \
        /usr/local/lib/libopencv_bioinspired.so.3.1 \
        /usr/local/lib/libopencv_calib3d.so.3.1 \
        /usr/local/lib/libopencv_ccalib.so.3.1 \
        /usr/local/lib/libopencv_core.so.3.1 \
        /usr/local/lib/libopencv_datasets.so.3.1 \
        /usr/local/lib/libopencv_dnn.so.3.1 \
        /usr/local/lib/libopencv_dpm.so.3.1 \
        /usr/local/lib/libopencv_face.so.3.1 \
        /usr/local/lib/libopencv_features2d.so.3.1 \
        /usr/local/lib/libopencv_flann.so.3.1 \
        /usr/local/lib/libopencv_fuzzy.so.3.1 \
        /usr/local/lib/libopencv_hdf.so.3.1 \
        /usr/local/lib/libopencv_highgui.so.3.1 \
        /usr/local/lib/libopencv_imgcodecs.so.3.1 \
        /usr/local/lib/libopencv_imgproc.so.3.1 \
        /usr/local/lib/libopencv_line_descriptor.so.3.1 \
        /usr/local/lib/libopencv_ml.so.3.1 \
        /usr/local/lib/libopencv_objdetect.so.3.1 \
        /usr/local/lib/libopencv_optflow.so.3.1 \
        /usr/local/lib/libopencv_photo.so.3.1 \
        /usr/local/lib/libopencv_plot.so.3.1 \
        /usr/local/lib/libopencv_reg.so.3.1 \
        /usr/local/lib/libopencv_rgbd.so.3.1 \
        /usr/local/lib/libopencv_saliency.so.3.1 \
        /usr/local/lib/libopencv_sfm.so.3.1 \
        /usr/local/lib/libopencv_shape.so.3.1 \
        /usr/local/lib/libopencv_stereo.so.3.1 \
        /usr/local/lib/libopencv_stitching.so.3.1 \
        /usr/local/lib/libopencv_structured_light.so.3.1 \
        /usr/local/lib/libopencv_superres.so.3.1 \
        /usr/local/lib/libopencv_surface_matching.so.3.1 \
        /usr/local/lib/libopencv_text.so.3.1 \
        /usr/local/lib/libopencv_tracking.so.3.1 \
        /usr/local/lib/libopencv_videoio.so.3.1 \
        /usr/local/lib/libopencv_video.so.3.1 \
        /usr/local/lib/libopencv_videostab.so.3.1 \
        /usr/local/lib/libopencv_viz.so.3.1 \
        /usr/local/lib/libopencv_xfeatures2d.so.3.1 \
        /usr/local/lib/libopencv_ximgproc.so.3.1 \
        /usr/local/lib/libopencv_xobjdetect.so.3.1 \
        /usr/local/lib/libopencv_xphoto.so.3.1
