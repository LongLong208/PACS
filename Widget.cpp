#include "Widget.h"
#include "./ui_Widget.h"

#include <QtDebug>
#include <QFileDialog>
#include <QMouseEvent>
#include <QMessageBox>

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
    connect(ui->segmentationBtn_2, &QPushButton::clicked, this, &Widget::segmentation2);
    connect(ui->openBtn, &QPushButton::clicked, this, &Widget::img_open);
    connect(ui->closeBtn, &QPushButton::clicked, this, &Widget::img_close);
    connect(ui->exportBtn, &QPushButton::clicked, this, &Widget::export_file);
    connect(ui->edgeDetectBtn, &QPushButton::clicked, this, &Widget::edgeDetect);
    connect(ui->threshold1Bar, &QSlider::valueChanged, this, &Widget::bar1Changed);
    connect(ui->threshold2Bar, &QSlider::valueChanged, this, &Widget::bar2Changed);
    connect(ui->regionGrowBar, &QSlider::valueChanged, this, &Widget::regionGrowBarChanged);
    connect(ui->compareBtn, &QPushButton::pressed, this, &Widget::compare);

    void (Widget::*ptr)(void) = &Widget::showImg;
    connect(ui->compareBtn, &QPushButton::released, this, ptr);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::setAvailable(bool enabled)
{
    ui->enhanceBtn->setEnabled(enabled);
    ui->sharpenBtn->setEnabled(enabled);
    ui->sharpenOpChoose->setEnabled(enabled);
    ui->blurBtn->setEnabled(enabled);
    ui->blurChoose->setEnabled(enabled);
    ui->segmentationBtn->setEnabled(enabled);
    ui->segmentationBtn_2->setEnabled(enabled);
    ui->openBtn->setEnabled(enabled);
    ui->closeBtn->setEnabled(enabled);
    ui->exportBtn->setEnabled(enabled);
    ui->threshold1Bar->setEnabled(enabled);
    ui->threshold2Bar->setEnabled(enabled);
    ui->edgeDetectBtn->setEnabled(enabled);
    ui->regionGrowBar->setEnabled(enabled);
    ui->compareBtn->setEnabled(enabled);
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

void Widget::showImg(cv::Mat &image)
{
    QImage qimg(image.data, 512, 512, QImage::Format_Grayscale8);
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
    QString fileName = QFileDialog::getOpenFileName(this, tr("打开dicom文件"), "C:\\Users\\lenovo\\Downloads\\Head\\", tr("dicom文件 (*.dcm)"));
    vtkSmartPointer<vtkDICOMImageReader> reader = vtkSmartPointer<vtkDICOMImageReader>::New();
    if (fileName == "")
    {
        // dicomread("C:\\Users\\lenovo\\Downloads\\dcms\\vhf.1643.dcm", img, reader);
        return;
    }
    dicomread(fileName.toStdString(), img, reader);
    memo.clear();
    cur = -1;
    first = last = size = 0;
    ui->redoBtn->setEnabled(false);
    ui->undoBtn->setEnabled(false);

    flip(img, img, 0);

    // img = imread(fileName.toStdString());
    // cvtColor(img, img, CV_RGB2GRAY);

    img = convertDicom(img);

    origin = img.clone();
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
        // 高斯模糊
        cv::GaussianBlur(img, img, Size(5, 5), 0.8, 0.8);
    case 1:
        // 3x3 均值滤波
        cv::blur(img, img, Size(3, 3));
        break;
    case 2:
        // 5x5 均值滤波
        cv::blur(img, img, Size(5, 5));
        break;
    case 3:
        // 3x3 中值滤波
        medianBlur(img, img, 3);
        break;
    case 4:
        // 5x5 中值滤波
        medianBlur(img, img, 5);
        break;
    default:
        break;
    }

    memorize();
    showImg();
}

// 区域生长
Mat RegionGrow(Mat srcImage, Point pt, int ch1Thres, int ch1LowerBind = 0, int ch1UpperBind = 255)
{
    Point pToGrowing;                                     //待生长点位置
    int pGrowValue = 0;                                   //待生长点灰度值
    Scalar pSrcValue = 0;                                 //生长起点灰度值
    Scalar pCurValue = 0;                                 //当前生长点灰度值
    Mat growImage = Mat::zeros(srcImage.size(), CV_8UC1); //创建一个空白区域，填充为黑色
    //生长方向顺序数据
    int DIR[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
    vector<Point> growPtVector;                 //生长点栈
    growPtVector.push_back(pt);                 //将生长点压入栈中
    growImage.at<uchar>(pt.y, pt.x) = 255;      //标记生长点
    pSrcValue = srcImage.at<uchar>(pt.y, pt.x); //记录生长点的灰度值

    while (!growPtVector.empty()) //生长栈不为空则生长
    {
        pt = growPtVector.back(); //取出一个生长点
        growPtVector.pop_back();

        //分别对八个方向上的点进行生长
        for (int i = 0; i < 8; ++i)
        {
            pToGrowing.x = pt.x + DIR[i][0];
            pToGrowing.y = pt.y + DIR[i][1];
            //检查是否是边缘点
            if (pToGrowing.x < 0 || pToGrowing.y < 0 ||
                pToGrowing.x > (srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
                continue;

            pGrowValue = growImage.at<uchar>(pToGrowing.y, pToGrowing.x); //当前待生长点的灰度值
            pSrcValue = srcImage.at<uchar>(pt.y, pt.x);
            if (pGrowValue == 0) //如果标记点还没有被生长
            {
                pCurValue = srcImage.at<uchar>(pToGrowing.y, pToGrowing.x);
                if (pCurValue[0] <= ch1UpperBind && pCurValue[0] >= ch1LowerBind)
                {
                    if (abs(pSrcValue[0] - pCurValue[0]) < ch1Thres) //在阈值范围内则生长
                    {
                        growImage.at<uchar>(pToGrowing.y, pToGrowing.x) = 255; //标记为白色
                        growPtVector.push_back(pToGrowing);                    //将下一个生长点压入栈中
                    }
                }
            }
        }
    }
    return growImage.clone();
}

// 边缘检测
void Widget::edgeDetect()
{
    Canny(img, img, ui->threshold1Bar->value(), ui->threshold2Bar->value());
    memorize();
    showImg();
}

// 分水岭分割
void Widget::segmentation()
{
    Mat temp;
    cvtColor(img, temp, COLOR_GRAY2RGB);

    //查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    Mat imageContours = Mat::zeros(img.size(), CV_8UC1); //轮廓
    Mat marks(img.size(), CV_32S);                       //Opencv分水岭第二个矩阵参数
    marks = Scalar::all(0);
    int index = 0;
    int compCount = 0;
    for (; index >= 0; index = hierarchy[index][0], compCount++)
    {
        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
        drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
        drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
    }

    watershed(temp, marks);

    marks.convertTo(img, CV_8UC1);

    memorize();
    showImg();
}

// 区域生长
void Widget::segmentation2()
{
    QRect posRect = ui->imageLabel->geometry();
    if (xpos < posRect.x() || xpos > posRect.x() + posRect.width() || ypos < posRect.y() || ypos > posRect.y() + posRect.height())
    {
        QMessageBox::warning(this, "错误", "请先选择种子点！");
        return;
    }
    img = RegionGrow(img, Point(xpos - posRect.x(), ypos - posRect.y()), ui->regionGrowBar->value());
    memorize();
    showImg();
}

// 开运算
void Widget::img_open()
{
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(img, img, kernel);
    dilate(img, img, kernel);
    memorize();
    showImg();
}

// 闭运算
void Widget::img_close()
{
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(img, img, kernel);
    erode(img, img, kernel);
    memorize();
    showImg();
}

// 导出
void Widget::export_file()
{
    QImage image(img.data, 512, 512, QImage::Format_Grayscale8);
    QString path = QFileDialog::getSaveFileName(this, tr("导出图片"), QDir::currentPath() + "\\untitled.jpg", tr("图象 (*.png *.jpg)"));
    image.save(path);
}

void Widget::bar1Changed()
{
    ui->label->setText(QString::number(ui->threshold1Bar->value()));
}

void Widget::bar2Changed()
{
    ui->label_2->setText(QString::number(ui->threshold2Bar->value()));
}

void Widget::mousePressEvent(QMouseEvent *event)
{
    xpos = event->x();
    ypos = event->y();
}

void Widget::regionGrowBarChanged()
{
    ui->label_3->setText(QString::number(ui->regionGrowBar->value()));
}

void Widget::compare()
{
    showImg(origin);
}
