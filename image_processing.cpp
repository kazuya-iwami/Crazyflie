#include "image_processing.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core_c.h>
#include "control.h"



using namespace std;
using namespace cv;

void BubSort(float arr[ ], int n);
int decodeMarker(IplImage* src,int& MarkerDirection);


void imageInit(){

    #define MARKER_SIZE (70)       /* マーカーの内側の1辺のサイズ[mm] */

    CvMat *intrinsic = (CvMat*)cvLoad("intrinsic.xml");
    CvMat *distortion = (CvMat*)cvLoad("distortion.xml");
    int i,j,k;

    CvMat object_points;
    CvMat image_points;

    CvMat *rotation = cvCreateMat (1, 3, CV_32FC1);
    CvMat *translation = cvCreateMat (1 , 3, CV_32FC1);

    //軸の座標生成用
    CvMat *srcPoints3D = cvCreateMat (4, 1, CV_32FC3);//元の3次元座標
    CvMat *dstPoints2D = cvCreateMat (4, 1, CV_32FC2);//画面に投影したときの2次元座標

    CvPoint3D32f baseMarkerPoints[4];
    //四角が物理空間上ではどの座標になるかを指定する。
    //コーナー      実際の座標(mm)
    //   X   Y     X    Y
    //   0   0   = 0    0
    //   0   1   = 0    20
    //   1   0   = 20   0
    baseMarkerPoints[0].x =(float) 0 * MARKER_SIZE;
    baseMarkerPoints[0].y =(float) 0 * MARKER_SIZE;
    baseMarkerPoints[0].z = 0.0;

    baseMarkerPoints[1].x =(float) 0 * MARKER_SIZE;
    baseMarkerPoints[1].y =(float) 1 * MARKER_SIZE;
    baseMarkerPoints[1].z = 0.0;

    baseMarkerPoints[2].x =(float) 1 * MARKER_SIZE;
    baseMarkerPoints[2].y =(float) 1 * MARKER_SIZE;
    baseMarkerPoints[2].z = 0.0;

    baseMarkerPoints[3].x =(float) 1 * MARKER_SIZE;
    baseMarkerPoints[3].y =(float) 0 * MARKER_SIZE;
    baseMarkerPoints[3].z = 0.0;

    //軸の基本座標を求める。
    for ( i=0;i<4;i++)
    {
        switch (i)
        {
            case 0:	srcPoints3D->data.fl[0]     =0;
                srcPoints3D->data.fl[1]     =0;
                srcPoints3D->data.fl[2]     =0;
                break;
            case 1:	srcPoints3D->data.fl[0+i*3] =(float)MARKER_SIZE;
                srcPoints3D->data.fl[1+i*3] =0;
                srcPoints3D->data.fl[2+i*3] =0;
                break;
            case 2:	srcPoints3D->data.fl[0+i*3] =0;
                srcPoints3D->data.fl[1+i*3] =(float)MARKER_SIZE;
                srcPoints3D->data.fl[2+i*3] =0;
                break;
            case 3:	srcPoints3D->data.fl[0+i*3] =0;
                srcPoints3D->data.fl[1+i*3] =0;
                srcPoints3D->data.fl[2+i*3] =-(float)MARKER_SIZE;;
                break;

        }
    }

    ///軸の準備　ここまで
    ////////////////////////////////////


    IplImage* image;
    IplImage* gsImage;
    IplImage* gsImageContour;

    CvCapture *capture = cvCaptureFromCAM(0);
    IplImage * capimg;

    capimg=cvQueryFrame(capture);

    double	process_time;
    image=cvCreateImage(cvGetSize(capimg),IPL_DEPTH_8U,3);
    cvCopy(capimg,image);

    gsImage=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    gsImageContour=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    char presskey;

    //フォントの設定
    CvFont dfont;
    float hscale      = 0.5f;
    float vscale      = 0.5f;
    float italicscale = 0.0f;
    int  thickness    = 1;
    char text[255] = "";
    cvInitFont (&dfont, CV_FONT_HERSHEY_SIMPLEX , hscale, vscale, italicscale, thickness, CV_AA);

    CvFont axisfont;
    float axhscale      = 0.8f;
    float axvscale      = 0.8f;
    cvInitFont (&axisfont, CV_FONT_HERSHEY_SIMPLEX , axhscale, axvscale, italicscale, thickness, CV_AA);


    //輪郭保存用のストレージを確保
    CvMemStorage *storage = cvCreateMemStorage (0);//輪郭用
    CvMemStorage *storagepoly = cvCreateMemStorage (0);//輪郭近似ポリゴン用

    CvSeq *firstcontour=NULL;
    CvSeq *polycontour=NULL;

    int contourCount;

    IplImage *marker_inside=cvCreateImage(cvSize(70,70),IPL_DEPTH_8U,1);
    IplImage *marker_inside_zoom=cvCreateImage(cvSize(marker_inside->width*2,marker_inside->height*2),IPL_DEPTH_8U,1);
    IplImage *tmp_img=cvCloneImage(marker_inside);

    CvMat *map_matrix;
    CvPoint2D32f src_pnt[4], dst_pnt[4], tmp_pnt[4];

    //マーカーの内側の変形先の形
    dst_pnt[0] = cvPoint2D32f (0, 0);
    dst_pnt[1] = cvPoint2D32f (marker_inside->width, 0);
    dst_pnt[2] = cvPoint2D32f (marker_inside->width, marker_inside->height);
    dst_pnt[3] = cvPoint2D32f (0, marker_inside->height);
    map_matrix = cvCreateMat (3, 3, CV_32FC1);

    //	cvNamedWindow ("marker_inside", CV_WINDOW_AUTOSIZE);
    //	cvNamedWindow ("inside", CV_WINDOW_AUTOSIZE);
    cvNamedWindow ("capture_image", CV_WINDOW_AUTOSIZE);


}


void imageProcess(cv::Mat &input, Drone *drone){

    cvClearMemStorage(storage);
    cvClearMemStorage(storagepoly);

    process_time = (double)cvGetTickCount();
    capimg=cvQueryFrame(capture);
    cvCopy(capimg,image);
    //グレースケール化
    cvCvtColor(image,gsImage,CV_BGR2GRAY);

    //平滑化
    cvSmooth(gsImage,gsImage,CV_GAUSSIAN,3);

    //二値化
    cvThreshold (gsImage, gsImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);//マーカーが浮き出る

    //反転
    cvNot(gsImage,gsImage);

    //輪郭を探します。Ncには探した輪郭の数が入ります。
    //CV_RETR_LISTに設定すると、見つけた輪郭がすべて同じレベルに入ります。。
    //first=輪郭１⇔輪郭２⇔輪郭３⇔輪郭４

    contourCount=0;

    //輪郭抽出
    cvCopy(gsImage,gsImageContour);
    contourCount=cvFindContours (gsImageContour, storage, &firstcontour, sizeof (CvContour), CV_RETR_CCOMP);

    //輪郭に近似しているポリゴンを求める（最小直線距離3ピクセルに設定）
    polycontour=cvApproxPoly(firstcontour,sizeof(CvContour),storagepoly,CV_POLY_APPROX_DP,3,1);

    for(CvSeq* c=polycontour;c!=NULL;c=c->h_next)
    {
        //外側の輪郭が大きすぎても小さすぎてもダメ。さらに四角形で無いとダメ　ということにする
        if((cvContourPerimeter(c)<2500)&&(cvContourPerimeter(c)>60)&&(c->total==4))
        {
            //四角形の中に四角形があればマーカーとする。
            if(c->v_next!=NULL)
            {
                if(c->v_next->total==4)
                {
                    int nearestindex=0;

                    CvSeq* c_vnext=c->v_next;
                    //			cvDrawContours(image,c,CV_RGB(255,255,0),CV_RGB(200,255,255),0);
                    //			cvDrawContours(image,c_vnext,CV_RGB(255,0,0),CV_RGB(0,255,255),0);


                    float xlist[4];
                    float ylist[4];
                    for(int n=0;n<4;n++)
                    {
                        CvPoint* p=CV_GET_SEQ_ELEM(CvPoint,c->v_next,n);


                        tmp_pnt[n].x=(float)p->x;
                        tmp_pnt[n].y=(float)p->y;
                        xlist[n]=(float)p->x;
                        ylist[n]=(float)p->y;
                    }

                    //四角の情報だけ渡す。どちらを向いているかはまだわからない
                    cvGetPerspectiveTransform (tmp_pnt, dst_pnt, map_matrix);

                    //マーカーの内側を正方形に変形させる
                    cvWarpPerspective (gsImage, marker_inside, map_matrix, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll (0));


                    //marker_inside（マーカーの内側だけを抽出し、正方形に透視変換したもの）
                    //を、マスク画像を指定して一時イメージにコピー。
                    //一時イメージに白い点が多数あれば、マスク画像と同じ方向を向いていることになる。

                    int notzeroCount=0;

                    int maxCount=0;
                    int markerDirection=0;//基本は0deg
                    cvResize(marker_inside,marker_inside_zoom);


                    //マーカーの番号を読み取る
                    int marker_number=decodeMarker(marker_inside,markerDirection);
                    sprintf(text,"%d",marker_number);
                    /*
                     decodeMarkerで方向がわかるようになったので、この辺はいらなくなった

                     cvCopy(marker_inside,tempmask,mask0);//cvCopy時にマスク画像を入れると、マスクの０じゃないところのみコピーされる。
                     notzeroCount=cvCountNonZero(tempmask);
                     if(maxCount<notzeroCount)
                     {
                     maxCount=notzeroCount;
                     markerDirection=0;
                     sprintf(text,"0deg", notzeroCount);

                     }
                     cvZero(tempmask);
                     cvCopy(marker_inside,tempmask,mask90);
                     notzeroCount=cvCountNonZero(tempmask);
                     if(maxCount<notzeroCount)
                     {
                     maxCount=notzeroCount;
                     markerDirection=90;
                     sprintf(text,"90deg");
                     }
                     cvZero(tempmask);
                     cvCopy(marker_inside,tempmask,mask180);
                     notzeroCount=cvCountNonZero(tempmask);
                     if(maxCount<notzeroCount)
                     {
                     maxCount=notzeroCount;
                     markerDirection=180;
                     sprintf(text,"180deg");
                     }
                     cvZero(tempmask);
                     cvCopy(marker_inside,tempmask,mask270);
                     notzeroCount=cvCountNonZero(tempmask);
                     if(maxCount<notzeroCount)
                     {
                     maxCount=notzeroCount;
                     markerDirection=270;
                     sprintf(text,"270deg");
                     }
                     cvPutText(marker_inside_zoom, text, cvPoint(70, 70), &dfont, cvScalarAll(255));

                     cvPutText(marker_inside_zoom, text, cvPoint(20, 120), &dfont, cvScalarAll(255));
                     cvZero(tempmask);
                     */

                    //		cvShowImage ("inside", marker_inside);
                    //			cvShowImage ("marker_inside", marker_inside_zoom);


                    //四角の向きを反映させる。

                    if(markerDirection==0)
                    {
                        src_pnt[0].x=tmp_pnt[0].x;
                        src_pnt[0].y=tmp_pnt[0].y;
                        src_pnt[1].x=tmp_pnt[3].x;
                        src_pnt[1].y=tmp_pnt[3].y;
                        src_pnt[2].x=tmp_pnt[2].x;
                        src_pnt[2].y=tmp_pnt[2].y;
                        src_pnt[3].x=tmp_pnt[1].x;
                        src_pnt[3].y=tmp_pnt[1].y;

                    }
                    if(markerDirection==90)
                    {
                        src_pnt[0].x=tmp_pnt[1].x;
                        src_pnt[0].y=tmp_pnt[1].y;
                        src_pnt[1].x=tmp_pnt[0].x;
                        src_pnt[1].y=tmp_pnt[0].y;
                        src_pnt[2].x=tmp_pnt[3].x;
                        src_pnt[2].y=tmp_pnt[3].y;
                        src_pnt[3].x=tmp_pnt[2].x;
                        src_pnt[3].y=tmp_pnt[2].y;


                    }

                    if(markerDirection==180)
                    {


                        src_pnt[0].x=tmp_pnt[2].x;
                        src_pnt[0].y=tmp_pnt[2].y;
                        src_pnt[1].x=tmp_pnt[1].x;
                        src_pnt[1].y=tmp_pnt[1].y;
                        src_pnt[2].x=tmp_pnt[0].x;
                        src_pnt[2].y=tmp_pnt[0].y;
                        src_pnt[3].x=tmp_pnt[3].x;
                        src_pnt[3].y=tmp_pnt[3].y;
                    }


                    if(markerDirection==270)
                    {
                        src_pnt[0].x=tmp_pnt[3].x;
                        src_pnt[0].y=tmp_pnt[3].y;
                        src_pnt[1].x=tmp_pnt[2].x;
                        src_pnt[1].y=tmp_pnt[2].y;
                        src_pnt[2].x=tmp_pnt[1].x;
                        src_pnt[2].y=tmp_pnt[1].y;
                        src_pnt[3].x=tmp_pnt[0].x;
                        src_pnt[3].y=tmp_pnt[0].y;
                    }
                    //		cvPutText(image,"0", cvPoint((int)src_pnt[0].x,(int)src_pnt[0].y), &dfont, CV_RGB(255, 0, 255));
                    //		cvPutText(image,"1", cvPoint((int)src_pnt[1].x,(int)src_pnt[1].y), &dfont, CV_RGB(255, 0, 255));



                    //マーカーのイメージ上での座標を設定。
                    cvInitMatHeader (&image_points, 4, 1, CV_32FC2, src_pnt);

                    //マーカーの基本となる座標を設定
                    cvInitMatHeader (&object_points, 4, 3, CV_32FC1, baseMarkerPoints);

                    //カメラの内部定数(intrinsticとdistortion)から、rotationとtranslationを求める
                    cvFindExtrinsicCameraParams2(&object_points,&image_points,intrinsic,distortion,rotation,translation);

                    //求めたものを使用して、現実空間上の座標が画面上だとどの位置に来るかを計算
                    cvProjectPoints2(srcPoints3D,rotation,translation,intrinsic,distortion,dstPoints2D);

                    //軸を描画
                    CvPoint startpoint;

                    CvPoint endpoint;

                    startpoint=cvPoint((int)dstPoints2D->data.fl[0], (int)dstPoints2D->data.fl[1]);

                    std::cout << "x:" << startpoint.x << " y:"<<startpoint.y << std::endl;
//                        for(j=1;j<4;j++)
//                        {
//                            endpoint=  cvPoint((int)dstPoints2D->data.fl[(j)*3],(int)dstPoints2D->data.fl[1+(j)*3]);
//
//                            if(j==1)
//                            {
//                                cvLine(image,startpoint,endpoint,CV_RGB(255,0,0),2,8,0);
//                                cvPutText(image, "X", endpoint, &axisfont,CV_RGB(255,0,0));
//                            }
//                            if(j==2)
//                            {
//                                cvLine(image,startpoint,endpoint,CV_RGB(0,255,0),2,8,0);
//                                cvPutText(image, "Y", endpoint, &axisfont,CV_RGB(0,255,0));
//                            }
//                            if(j==3)
//                            {
//                                cvLine(image,startpoint,endpoint,CV_RGB(0,0,255),2,8,0);
//                                cvPutText(image, "Z", endpoint, &axisfont,CV_RGB(0,0,255));
//                            }
//                        }
//                        //マーカーの番号を描画
//                        cvPutText(image, text,cvPoint((int)(dstPoints2D->data.fl[3]+dstPoints2D->data.fl[6])/2,
//                                                      (int)(dstPoints2D->data.fl[4]+dstPoints2D->data.fl[7])/2)
//                                  , &axisfont, CV_RGB(255, 255, 100));
                }
            }

        }
    }

    process_time = (double)cvGetTickCount()-process_time;

    sprintf(text,"process_time %gms", process_time/(cvGetTickFrequency()*1000.));
    cvPutText(image, "http://playwithopencv.blogspot.com", cvPoint(10, 20), &dfont, CV_RGB(255, 255, 255));
    cvPutText(image, text, cvPoint(10, 40), &dfont, CV_RGB(255, 255, 255));
    cvShowImage("capture_image",image);


    presskey=cvWaitKey (50);
    if(presskey==27)break;
}

void imageRelease(){

}