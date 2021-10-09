#include "Widget.h"
#include "./ui_Widget.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

Widget::Widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);

    Mat img = imread("D:\\DICOM.jpg");
    cv::imshow("image", img);
}

Widget::~Widget()
{
    delete ui;
}
