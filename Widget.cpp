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

Widget::Widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
    memo.resize(memoSize);
    size = first = last = 0;
    cur = -1;
    ui->redoBtn->setEnabled(false);
    ui->undoBtn->setEnabled(false);
    connect(ui->readBtn, &QPushButton::clicked, this, &Widget::readDicom);
    connect(ui->enhanceBtn, &QPushButton::clicked, this, &Widget::enhance);
    connect(ui->undoBtn, &QPushButton::clicked, this, &Widget::undo);
    connect(ui->redoBtn, &QPushButton::clicked, this, &Widget::redo);
}

Widget::~Widget()
{
    delete ui;
}

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

void Widget::showImg()
{
    QImage qimg(img.data, 512, 512, QImage::Format_Grayscale8);
    ui->imageLabel->setPixmap(QPixmap::fromImage(qimg));
}

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

void Widget::readDicom()
{
    memo.clear();
    cur = -1;
    first = last = size = 0;
    ui->redoBtn->setEnabled(false);
    ui->undoBtn->setEnabled(false);
    QString fileName = QFileDialog::getOpenFileName(this, tr("打开dicom文件"), QDir::currentPath(), tr("dicom文件 (*.dcm)"));
    vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
    if (fileName == "")
        dicomread("C:\\Users\\lenovo\\Downloads\\dcms\\vhf.1643.dcm", img, reader);
    else
        dicomread(fileName.toStdString(), img, reader);
    flip(img, img, 0);
    // cout << img.channels() << "  " << img.size() << endl;
    // img.convertTo(img, CV_32F, 1.0 / 255, 0);
    // imshow("image", img);
    img = convertDicom(img);
    // imshow("image", img);
    memorize();
    showImg();
}

void Widget::enhance()
{
    switch (ui->enhanceChoose->currentIndex())
    {
    case 0:
        // 直方图均衡化
        equalizeHist(img, img);
        break;

    case 1:
        // 拉普拉斯
        Mat temp;
        Mat kernel = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        filter2D(img, temp, CV_8U, kernel);
        img = img - temp;
    }
    memorize();
    showImg();
}
