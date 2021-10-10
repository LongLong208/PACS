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
    const int memoSize = 10;
    vector<cv::Mat> memo;
    int cur, first, last, size;
    cv::Mat img;
    void showImg();

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    void memorize();

public slots:
    void readDicom();
    void enhance();
    void undo();
    void redo();

private:
    Ui::Widget *ui;
};
#endif // WIDGET_H
