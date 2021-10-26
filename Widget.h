#ifndef WIDGET_H
#define WIDGET_H

#include <vector>

#include <QWidget>

#include <opencv2/core/core.hpp>

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui
{
    class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

protected:
    const int memoSize = 100;
    vector<cv::Mat> memo;
    int cur, first, last, size;
    cv::Mat img;
    cv::Mat origin;
    void showImg();
    void showImg(cv::Mat &image);
    void setAvailable(bool);
    int xpos;
    int ypos;

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    void memorize();

public slots:
    void readDicom();
    void enhance();
    void undo();
    void redo();
    void sharpen();
    void blur();
    void segmentation();
    void segmentation2();
    void img_open();
    void img_close();
    void export_file();
    void bar1Changed();
    void bar2Changed();
    void regionGrowBarChanged();
    void edgeDetect();
    void compare();
    void mousePressEvent(QMouseEvent *) override;

private:
    Ui::Widget *ui;
};
#endif // WIDGET_H
