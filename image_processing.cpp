#include "image_processing.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include "control.h"



using namespace std;
using namespace cv;

CvConDensation *con;
CvMat *lowerBound;
CvMat *upperBound;

CvCapture *capture;
IplImage *image;
IplImage *capimg;

int minH = 0, maxH = 255;
int minS = 0, maxS = 255;
int minV = 0, maxV = 255;

void imageInit(){

    // パーティクルフィルタ
    con = cvCreateConDensation(4, 0, 3000);

    capture = cvCaptureFromCAM(0);
    capimg=cvQueryFrame(capture);
    image=cvCreateImage(cvGetSize(capimg),IPL_DEPTH_8U,3);
    cvCopy(capimg,image);

    // フィルタの設定
    CvMat *lowerBound = cvCreateMat(4, 1, CV_32FC1);
    CvMat *upperBound = cvCreateMat(4, 1, CV_32FC1);
    cvmSet(lowerBound, 0, 0, 0);
    cvmSet(lowerBound, 1, 0, 0);
    cvmSet(lowerBound, 2, 0, -10);
    cvmSet(lowerBound, 3, 0, -10);
    cvmSet(upperBound, 0, 0, image->width);
    cvmSet(upperBound, 1, 0, image->height);
    cvmSet(upperBound, 2, 0, 10);
    cvmSet(upperBound, 3, 0, 10);

    // 初期化
    cvConDensInitSampleSet(con, lowerBound, upperBound);

    // 等速直線運動モデル
    con->DynamMatr[0]  = 1.0; con->DynamMatr[1]  = 0.0; con->DynamMatr[2]  = 1.0; con->DynamMatr[3]  = 0.0;
    con->DynamMatr[4]  = 0.0; con->DynamMatr[5]  = 1.0; con->DynamMatr[6]  = 0.0; con->DynamMatr[7]  = 1.0;
    con->DynamMatr[8]  = 0.0; con->DynamMatr[9]  = 0.0; con->DynamMatr[10] = 1.0; con->DynamMatr[11] = 0.0;
    con->DynamMatr[12] = 0.0; con->DynamMatr[13] = 0.0; con->DynamMatr[14] = 0.0; con->DynamMatr[15] = 1.0;

    // ノイズパラメータの設定
    cvRandInit(&(con->RandS[0]), -25, 25, (int)cvGetTickCount());
    cvRandInit(&(con->RandS[1]), -25, 25, (int)cvGetTickCount());
    cvRandInit(&(con->RandS[2]),  -5,  5, (int)cvGetTickCount());
    cvRandInit(&(con->RandS[3]),  -5,  5, (int)cvGetTickCount());

    // 閾値
    int minH = 0, maxH = 255;
    int minS = 0, maxS = 255;
    int minV = 0, maxV = 255;

    // ウィンドウ
    cvNamedWindow("binalized");
    cvCreateTrackbar("H max", "binalized", &maxH, 255);
    cvCreateTrackbar("H min", "binalized", &minH, 255);
    cvCreateTrackbar("S max", "binalized", &maxS, 255);
    cvCreateTrackbar("S min", "binalized", &minS, 255);
    cvCreateTrackbar("V max", "binalized", &maxV, 255);
    cvCreateTrackbar("V min", "binalized", &minV, 255);
    cvResizeWindow("binalized", 0, 0);

    cvNamedWindow ("capture_image", CV_WINDOW_AUTOSIZE);


}


void imageProcess(Drone *drone){

    drone->captured_flag=false;

    capimg=cvQueryFrame(capture);
    image=cvCreateImage(cvGetSize(capimg),IPL_DEPTH_8U,3);
    cvCopy(capimg,image);

    // HSVに変換
    IplImage *hsv = cvCloneImage(image);
    cvCvtColor(image, hsv, CV_RGB2HSV_FULL);

    // 2値化画像
    IplImage *binalized = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

    // 2値化
    CvScalar lower = cvScalar(minH, minS, minV);
    CvScalar upper = cvScalar(maxH, maxS, maxV);
    cvInRangeS(hsv, lower, upper, binalized);

    // 結果を表示
    cvShowImage("binalized", binalized);

    // ノイズの除去
    cvMorphologyEx(binalized, binalized, NULL, NULL, CV_MOP_CLOSE);

    // 輪郭検出
    CvSeq *contour = NULL, *maxContour = NULL;
    CvMemStorage *contourStorage = cvCreateMemStorage();
    cvFindContours(binalized, contourStorage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // 一番大きな輪郭を抽出
    double max_area = 0.0;
    while (contour) {
        double area = fabs(cvContourArea(contour));
        if ( area > max_area) {
            maxContour = contour;
            max_area = area;
        }
        contour = contour->h_next;
    }

    // 検出できた
    if (maxContour) {
        // 輪郭の描画
        cvZero(binalized);
        cvDrawContours(binalized, maxContour, cvScalarAll(255), cvScalarAll(255), 0, CV_FILLED);

        // 重心を求める
        CvMoments moments;
        cvMoments(binalized, &moments, 1);
        int my = (int)(moments.m01/moments.m00);
        int mx = (int)(moments.m10/moments.m00);
        cvCircle(image, cvPoint(mx, my), 10, CV_RGB(255,0,0));

        // 各パーティクルの尤度（ゆうど）を求める
        for (int i = 0; i < con->SamplesNum; i++) {
            // サンプル
            float x = (con->flSamples[i][0]);
            float y = (con->flSamples[i][1]);

            // サンプルが画像範囲内にある
            if (x > 0 && x < image->width && y > 0 && y < image->height) {
                // ガウス分布とみなす
                double sigma = 50.0;
                double dist = hypot(x - mx, y - my);    // 重心に近いほど尤度が高い
                con->flConfidence[i] = 1.0 / (sqrt (2.0 * CV_PI) * sigma) * expf (-dist*dist / (2.0 * sigma*sigma));
            }
            else con->flConfidence[i] = 0.0;
            cvCircle(image, cvPointFrom32f(cvPoint2D32f(x, y)), 3, CV_RGB(0,128,con->flConfidence[i] * 50000));
        }
    }

    // 更新
    cvConDensUpdateByTime(con);

    // 重み付き平均用
    double sumX = 0, sumY = 0, sumConf = 0;
    for (int i = 0; i < con->SamplesNum; i++) {
        sumX += con->flConfidence[i] * con->flSamples[i][0];
        sumY += con->flConfidence[i] * con->flSamples[i][1];
        sumConf += con->flConfidence[i];
    }

    // 推定値を計算
    if (sumConf > 0.0) {
        float x = sumX / sumConf;
        float y = sumY / sumConf;
        cvCircle(image, cvPointFrom32f(cvPoint2D32f(x, y)), 10, CV_RGB(0,255,0));
    }

    // 表示
    cvShowImage("capture_image", image);

    // メモリ解放
    cvReleaseImage(&hsv);
    cvReleaseImage(&binalized);
    cvReleaseMemStorage(&contourStorage);

}

void imageRelease(){
    cvReleaseMat(&lowerBound);
    cvReleaseMat(&upperBound);
    cvReleaseConDensation(&con);
}

