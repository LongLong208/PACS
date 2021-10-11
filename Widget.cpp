#include "Widget.h"
#include "./ui_Widget.h"

#include <QtDebug>
#include <QFileDialog>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkSmartPointer.h>
#include <vtkImageViewer2.h>
#include <vtkImageCast.h>
#include <vtkDICOMImageReader.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkImageData.h>
#include <vtkCoordinate.h>

using namespace cv;
using namespace std;

// dicom 读取
void dicomread(string inputFilename, Mat &img, vtkSmartPointer<vtkDICOMImageReader> &reader)
{
    img.create(512, 512, CV_32SC1);

    vtkSmartPointer<vtkImageCast> imageCast = vtkSmartPointer<vtkImageCast>::New();

    reader->SetFileName(inputFilename.c_str());

    reader->Update();

    imageCast->SetInputConnection(reader->GetOutputPort());
    imageCast->SetOutputScalarTypeToInt();
    imageCast->Update();

    // 图像的基本信息
    int dims[3];
    reader->GetOutput()->GetDimensions(dims);

    //图像的像素值
    for (int k = 0; k < dims[2]; k++)
    {
        for (int j = 0; j < dims[1]; j++)
        {
            for (int i = 0; i < dims[0]; i++)
            {
                int *pixel =
                    (int *)(imageCast->GetOutput()->GetScalarPointer(i, j, k)); // 第i列第j行的像素值
                img.at<int>(j, i) = int(*pixel);                                // 第j行第i列的像素值
            }
        }
    }
}

// 灰度范围修正
Mat convertDicom(const Mat &I)
{
    Mat ret = I;
    double max = 0, min = 0;
    minMaxIdx(ret, &min, &max);
    // ret.convertTo(ret, CV_64FC1, 1.0 / (max - min));
    ret.convertTo(ret, CV_64FC1, 1);
    double factor = 255 / (max - min);
    for (int i = 0; i < ret.rows; i++)
    {
        for (int j = 0; j < ret.cols; j++)
        {
            ret.at<double>(i, j) = factor * (ret.at<double>(i, j) - min);
        }
    }
    ret.convertTo(ret, CV_8U);
    return ret;
}

// 显示直方图
void printHist(Mat &img)
{
    Mat hist;
    int histSize = 255;
    float range[] = {0, 255};
    const float *histRange = range;

    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

    int hist_w = 400;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
    imshow("calcHist Demo", histImage);
    waitKey(0);
}

void Widget::setAvailable(bool enabled)
{
    ui->enhanceBtn->setEnabled(enabled);
    557nui ui->sharpenBtn->setEnabled(enabled);
    ui->sharpenOpChoose->setEnabled(enabled);
    ui->blurBtn->setEnabled(enabled);
    ui->blurChoose->setEnabled(enabled);
    ui->segmentationBtn->setEnabled(enabled);
}

Widget::Widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
    memo.resize(memoSize);
    size = first = last = 0;
    cur = -1;
    ui->redoBtn->setEnabled(false);
    ui->undoBtn->setEnabled(false);
    setAvailable(false);
    connect(ui->readBtn, &QPushButton::clicked, this, &Widget::readDicom);
    connect(ui->undoBtn, &QPushButton::clicked, this, &Widget::undo);
    connect(ui->redoBtn, &QPushButton::clicked, this, &Widget::redo);
    connect(ui->enhanceBtn, &QPushButton::clicked, this, &Widget::enhance);
    connect(ui->sharpenBtn, &QPushButton::clicked, this, &Widget::sharpen);
    connect(ui->blurBtn, &QPushButton::clicked, this, &Widget::blur);
    connect(ui->segmentationBtn, &QPushButton::clicked, this, &Widget::segmentation);
}

Widget::~Widget()
{
    delete ui;
}

// 撤销
void Widget::undo()
{
    cur = (memoSize + cur - 1) % memoSize;
    img = memo[cur].clone();
    ui->redoBtn->setEnabled(true);
    if (cur == first)
        ui->undoBtn->setEnabled(false);
    showImg();
    // qDebug() << cur;
}

// 重做
void Widget::redo()
{
    cur = (cur + 1) % memoSize;
    img = memo[cur].clone();
    ui->undoBtn->setEnabled(true);
    if ((cur + 1) % memoSize == last)
        ui->redoBtn->setEnabled(false);
    showImg();
    // qDebug() << cur;
}

// 显示当前图象
void Widget::showImg()
{
    QImage qimg(img.data, 512, 512, QImage::Format_Grayscale8);
    ui->imageLabel->setPixmap(QPixmap::fromImage(qimg));
}

// 记录历史图象
void Widget::memorize()
{
    if ((cur + 1) % memoSize == last)
    {
        memo[last] = img.clone();
        if (cur != -1 && last == first)
        {
            first = (first + 1) % memoSize;
        }
        cur = last;
        last = (last + 1) % memoSize;
    }
    else
    {
        cur = (cur + 1) % memoSize;
        last = (cur + 1) % memoSize;
        memo[cur] = img.clone();
        ui->redoBtn->setEnabled(false);
    }
    if (cur != first)
    {
        // imshow("aa", memo[0]);
        ui->undoBtn->setEnabled(true);
    }
    // qDebug() << cur;
}

// 读取图象
void Widget::readDicom()
{
    memo.clear();
    cur = -1;
    first = last = size = 0;
    ui->redoBtn->setEnabled(false);
    ui->undoBtn->setEnabled(false);
    QString fileName = QFileDialog::getOpenFileName(this, tr("打开dicom文件"), QDir::currentPath(), tr("dicom文件 (*.dcm), 所有文件 (*.*)"));
    vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
    if (fileName == "")
        dicomread("C:\\Users\\lenovo\\Downloads\\dcms\\vhf.1643.dcm", img, reader);
    else
        dicomread(fileName.toStdString(), img, reader);
    flip(img, img, 0);

    // img = imread(fileName.toStdString());
    // cvtColor(img, img, CV_RGB2GRAY);

    img = convertDicom(img);
    memorize();
    showImg();

    setAvailable(true);
}

// 图象增强
void Widget::enhance()
{
    // printHist(img);
    // 直方图均衡化
    equalizeHist(img, img);
    // printHist(img);
    memorize();
    showImg();
}

// 锐化
void Widget::sharpen()
{
    // 拉普拉斯
    Mat temp;
    Mat kernel;

    switch (ui->sharpenOpChoose->currentIndex())
    {
    case 0:
        kernel = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        break;
    case 1:
        kernel = (Mat_<float>(5, 5) << 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, -16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0);
        break;
    default:
        kernel = (Mat_<float>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
        break;
    }

    filter2D(img, temp, CV_8UC1, kernel);
    img = img - temp;

    memorize();
    showImg();
}

// 模糊、滤波
void Widget::blur()
{
    switch (ui->blurChoose->currentIndex())
    {
    case 0:
        // 3x3 均值滤波
        cv::blur(img, img, Size(3, 3));
        break;
    case 1:
        // 5x5 均值滤波
        cv::blur(img, img, Size(5, 5));
        break;
    case 2:
        // 3x3 中值滤波
        medianBlur(img, img, 3);
        break;
    case 3:
        // 5x5 中值滤波
        medianBlur(img, img, 5);
        break;
    default:
        break;
    }

    memorize();
    showImg();
}

// 图象分割
void Widget::segmentation()
{
    Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    filter2D(img, img, CV_8UC1, kernel);
    memorize();
    showImg();
}
