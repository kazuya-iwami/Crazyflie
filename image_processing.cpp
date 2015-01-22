
#include "image_processing.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include "control.h"



using namespace std;
using namespace cv;

char text[255] = "";
CvFont dfont;

double	process_time;


CvCapture *capture;
IplImage *image;
IplImage *capimg;

int minH, maxH ;
int minS, maxS ;
int minV, maxV ;

void imageInit(){

    capture = cvCaptureFromCAM(0);


    // 閾値
    minH = 129, maxH = 150;//オレンジに合わせた
    minS = 127, maxS = 255;
    minV = 0, maxV = 255;

    //フォントの設定

    float hscale      = 0.5f;
    float vscale      = 0.5f;
    float italicscale = 0.0f;
    int  thickness    = 1;

    cvInitFont (&dfont, CV_FONT_HERSHEY_SIMPLEX , hscale, vscale, italicscale, thickness, CV_AA);




}

void imageProcess(Drone *drone,double battery_level){

    process_time = (double)cvGetTickCount();

    image=cvQueryFrame(capture);

    // HSVに変換
    IplImage *hsv = cvCloneImage(image);
    cvCvtColor(image, hsv, CV_RGB2HSV_FULL);

    // 2値化画像
    IplImage *binalized = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

    // 2値化
    CvScalar lower = cvScalar(minH, minS, minV);
    CvScalar upper = cvScalar(maxH, maxS, maxV);
    cvInRangeS(hsv, lower, upper, binalized);

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
    if (maxContour && max_area > 150) {
        // 輪郭の描画
        cvZero(binalized);
        cvDrawContours(binalized, maxContour, cvScalarAll(255),cvScalarAll(255),  -1, CV_FILLED, 8);

        // 重心を求める
        CvMoments moments;
        cvMoments(binalized, &moments, 1);
        int my = (int)(moments.m01/moments.m00);
        int mx = (int)(moments.m10/moments.m00);
        cvCircle(image, cvPoint(mx, my), 10, CV_RGB(255,0,0));

        drone->cur_pos.x=mx;
        drone->cur_pos.y=my;
        drone->captured_flag = true;

    }else{
        drone->captured_flag = false;
    }


    cvCircle(image, cvPoint(drone->dst_pos.x,drone->dst_pos.y), 10, CV_RGB(0,0,255));//目標地点描写

    process_time = (double)cvGetTickCount()-process_time;

    sprintf(text,"process_time %gms", process_time/(cvGetTickFrequency()*1000));
    cvPutText(image, text, cvPoint(10, 40), &dfont, CV_RGB(255, 255, 255));
    sprintf(text,"battery_level %3fV/5V",battery_level);
    cvPutText(image, text, cvPoint(10, 20), &dfont, CV_RGB(255, 255, 255));

    // 表示
    cvShowImage("capture_image", image);

    // メモリ解放
    cvReleaseImage(&hsv);
    cvReleaseImage(&binalized);
    cvReleaseMemStorage(&contourStorage);

}

void imageRelease(){
    cvDestroyWindow ("capture_image");
    cvReleaseImage (&image);
    cvReleaseImage (&capimg);
    cvReleaseCapture (&capture);
}

