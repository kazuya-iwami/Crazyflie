#include "image_processing.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "control.h"


using namespace std;
using namespace cv;

int i, c;
double w = 0.0, h = 0.0;
CvCapture *capture;
IplImage *frame;
IplImage *image;

int n_stat;
int n_particle;
CvConDensation *cond;
CvMat *lowerBound;
CvMat *upperBound;

int xx, yy;


// (1)尤度の計算を行なう関数
float
calc_likelihood(IplImage *img, int x, int y) {
    float b, g, r;
    float dist = 0.0, sigma = 50.0;

    b = img->imageData[img->widthStep * y + x * 3];       //B
    g = img->imageData[img->widthStep * y + x * 3 + 1];   //G
    r = img->imageData[img->widthStep * y + x * 3 + 2];   //R
    dist = sqrt(b * b + g * g + (255.0-r)*(255.0-r));

    return 1.0 / (sqrt(2.0 * CV_PI) * sigma) * expf(-dist * dist / (2.0 * sigma * sigma));
}


void imageInit() {
    w = 0.0, h = 0.0;
    //capture = 0;
    //frame = 0;

    n_stat = 4;
    n_particle = 4000;
    cond = 0;
    lowerBound = 0;
    upperBound = 0;


    capture = cvCreateCameraCapture(0);
    if(!capture){
        cout << "err" << endl;
    }

    // (3)１フレームキャプチャし，キャプチャサイズを取得する．
    frame = cvQueryFrame(capture);
    image=cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,3);
    cvCopy(frame,image);
    w = image->width;
    h = image->height;

    cvNamedWindow("Condensation", CV_WINDOW_AUTOSIZE);

    // (4)Condensation構造体を作成する．
    cond = cvCreateConDensation(n_stat, 0, n_particle);

    // (5)状態ベクトル各次元の取りうる最小値・最大値を指定する．
    lowerBound = cvCreateMat(4, 1, CV_32FC1);
    upperBound = cvCreateMat(4, 1, CV_32FC1);

    cvmSet(lowerBound, 0, 0, 0.0);
    cvmSet(lowerBound, 1, 0, 0.0);
    cvmSet(lowerBound, 2, 0, -10.0);
    cvmSet(lowerBound, 3, 0, -10.0);
    cvmSet(upperBound, 0, 0, w);
    cvmSet(upperBound, 1, 0, h);
    cvmSet(upperBound, 2, 0, 10.0);
    cvmSet(upperBound, 3, 0, 10.0);

    // (6)Condensation構造体を初期化する
    cvConDensInitSampleSet(cond, lowerBound, upperBound);

    // (7)ConDensationアルゴリズムにおける状態ベクトルのダイナミクスを指定する
    cond->DynamMatr[0] = 1.0;
    cond->DynamMatr[1] = 0.0;
    cond->DynamMatr[2] = 1.0;
    cond->DynamMatr[3] = 0.0;
    cond->DynamMatr[4] = 0.0;
    cond->DynamMatr[5] = 1.0;
    cond->DynamMatr[6] = 0.0;
    cond->DynamMatr[7] = 1.0;
    cond->DynamMatr[8] = 0.0;
    cond->DynamMatr[9] = 0.0;
    cond->DynamMatr[10] = 1.0;
    cond->DynamMatr[11] = 0.0;
    cond->DynamMatr[12] = 0.0;
    cond->DynamMatr[13] = 0.0;
    cond->DynamMatr[14] = 0.0;
    cond->DynamMatr[15] = 1.0;

    // (8)ノイズパラメータを再設定する．
    cvRandInit(&(cond->RandS[0]), -25, 25, (int) cvGetTickCount());
    cvRandInit(&(cond->RandS[1]), -25, 25, (int) cvGetTickCount());
    cvRandInit(&(cond->RandS[2]), -5, 5, (int) cvGetTickCount());
    cvRandInit(&(cond->RandS[3]), -5, 5, (int) cvGetTickCount());


}


void imageProcess(Drone *drone) {


    drone->captured_flag = false;
    frame = cvQueryFrame(capture);
    cvCopy(frame,image);


    // (9)各パーティクルについて尤度を計算する．
    for (i = 0; i < n_particle; i++) {
        xx = (int) (cond->flSamples[i][0]);
        yy = (int) (cond->flSamples[i][1]);
        if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
            cond->flConfidence[i] = 0.0;
        }
        else {
            cond->flConfidence[i] = calc_likelihood(frame, xx, yy);
            cvCircle(image, cvPoint(xx, yy), 2, CV_RGB (0, 0, 255), -1);
        }
    }



    // (10)次のモデルの状態を推定する
    cvConDensUpdateByTime(cond);
    cvShowImage("Condensation", image);

}

void imageRelease() {
    cvDestroyWindow("Condensation");
    cvReleaseImage(&frame);
    cvReleaseCapture(&capture);
    cvReleaseImage(&image);
    cvReleaseConDensation(&cond);
    cvReleaseMat(&lowerBound);
    cvReleaseMat(&upperBound);
}

