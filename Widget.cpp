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
    connect(ui->readBtn, &QPushButton::clicked, this, &Widget::readDicom);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::readDicom()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("打开dicom文件"), QDir::currentPath(), tr("dicom文件 (*.dcm)"));

    Mat img;
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
    QImage qimg(img.data, 512, 512, QImage::Format_Grayscale8);
    ui->imageLabel->setPixmap(QPixmap::fromImage(qimg));
}
