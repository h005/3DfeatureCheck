QT += core
QT -= gui

CONFIG += c++11

TARGET = lineSegmentDetection
CONFIG += console


SOURCES += main.cpp \
    lsd.cpp \
    vpdetection.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \

LIBS += -L/usr/local/lib \
        -lopencv_aruco \
        -lopencv_bgsegm \
        -lopencv_bioinspired \
        -lopencv_calib3d \
        -lopencv_ccalib \
        -lopencv_core \
        -lopencv_datasets \
        -lopencv_dnn \
        -lopencv_dpm \
        -lopencv_face \
        -lopencv_features2d \
        -lopencv_flann \
        -lopencv_freetype \
        -lopencv_fuzzy \
        -lopencv_hdf \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_line_descriptor \
        -lopencv_ml \
        -lopencv_objdetect \
        -lopencv_optflow \
        -lopencv_phase_unwrapping \
        -lopencv_photo \
        -lopencv_plot \
        -lopencv_reg \
        -lopencv_rgbd \
        -lopencv_saliency \
#        -lopencv_sfm \
        -lopencv_shape \
        -lopencv_stereo \
        -lopencv_stitching \
        -lopencv_structured_light \
        -lopencv_superres \
        -lopencv_surface_matching \
        -lopencv_text \
        -lopencv_tracking \
        -lopencv_videoio \
        -lopencv_video \
        -lopencv_videostab \
        -lopencv_viz \
        -lopencv_xfeatures2d \
        -lopencv_ximgproc \
        -lopencv_xobjdetect \
        -lopencv_xphoto \

HEADERS += \
    lsd.h \
    vpdetection.h

