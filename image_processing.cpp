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


CvMat *intrinsic;
CvMat *distortion;

CvMat object_points;
CvMat image_points;

CvMat *rotation;
CvMat *translation;

char text[255] = "";
CvFont dfont;

//軸の座標生成用
CvMat *srcPoints3D ;//元の3次元座標
CvMat *dstPoints2D;//画面に投影したときの2次元座標

CvPoint3D32f baseMarkerPoints[4];

CvMemStorage *storage;
CvMemStorage *storagepoly;
double	process_time;

IplImage* image;
IplImage* gsImage;
IplImage* gsImageContour;

CvCapture *capture;
IplImage * capimg;

CvSeq *firstcontour=NULL;
CvSeq *polycontour=NULL;

int contourCount;


CvMat *map_matrix;
CvPoint2D32f src_pnt[4], dst_pnt[4], tmp_pnt[4];

IplImage *marker_inside;
IplImage *marker_inside_zoom;
IplImage *tmp_img;

void imageInit(){

    #define MARKER_SIZE (70)       /* マーカーの内側の1辺のサイズ[mm] */

    intrinsic = (CvMat*)cvLoad("intrinsic.xml");
    distortion = (CvMat*)cvLoad("distortion.xml");
    int i,j,k;

    rotation = cvCreateMat (1, 3, CV_32FC1);
    translation = cvCreateMat (1 , 3, CV_32FC1);

    //軸の座標生成用
    srcPoints3D = cvCreateMat (4, 1, CV_32FC3);//元の3次元座標
    dstPoints2D = cvCreateMat (4, 1, CV_32FC2);//画面に投影したときの2次元座標

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


    capture = cvCaptureFromCAM(0);
    if(!capture){
        cout << "err" << endl;
    }

    capimg=cvQueryFrame(capture);


    image=cvCreateImage(cvGetSize(capimg),IPL_DEPTH_8U,3);
    cvCopy(capimg,image);

    gsImage=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);
    gsImageContour=cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,1);


    //フォントの設定

    float hscale      = 0.5f;
    float vscale      = 0.5f;
    float italicscale = 0.0f;
    int  thickness    = 1;

    cvInitFont (&dfont, CV_FONT_HERSHEY_SIMPLEX , hscale, vscale, italicscale, thickness, CV_AA);

    CvFont axisfont;
    float axhscale      = 0.8f;
    float axvscale      = 0.8f;
    cvInitFont (&axisfont, CV_FONT_HERSHEY_SIMPLEX , axhscale, axvscale, italicscale, thickness, CV_AA);


    //輪郭保存用のストレージを確保
    storage = cvCreateMemStorage (0);//輪郭用
    storagepoly = cvCreateMemStorage (0);//輪郭近似ポリゴン用

    firstcontour=NULL;
    polycontour=NULL;

    marker_inside=cvCreateImage(cvSize(70,70),IPL_DEPTH_8U,1);
    marker_inside_zoom=cvCreateImage(cvSize(marker_inside->width*2,marker_inside->height*2),IPL_DEPTH_8U,1);
    tmp_img=cvCloneImage(marker_inside);


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


void imageProcess(Drone &drone){

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
                    drone.cur_pos.x=startpoint.x;
                    drone.cur_pos.y=startpoint.y;

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

}

void imageRelease(){

    cvReleaseImage(&image);
    cvReleaseImage(&gsImage);
    cvReleaseImage(&gsImageContour);
    cvReleaseImage(&marker_inside);
    cvReleaseImage(&tmp_img);

    cvReleaseMat (&map_matrix);
    cvReleaseMemStorage(&storagepoly);
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&image);
    cvReleaseImage(&gsImage);

    cvDestroyWindow("capture_image");

}




void BubSort(float arr[ ], int n)
{
    int i, j;
    float temp;

    for (i = 0; i < n - 1; i++)
    {
        for (j = n - 1; j > i; j--)
        {
            if (arr[j - 1] > arr[j])
            {  /* 前の要素の方が大きかったら */
                temp = arr[j];        /* 交換する */
                arr[j] = arr[j - 1];
                arr[j - 1]= temp;
            }
        }
    }
}


//マーカーの2次元コードを読み取る
//
int decodeMarker(IplImage* src,int& MarkerDirection)
{
    if(src->nChannels!=1)
    {
        return -1;
    }

    if(src->width!=src->height)
    {
        return -1;
    }


    //マーカーの内側のドットを解析する。
    //左上が基準ドット。
    //×で表示したところが白くなっているか黒くなっているかを判断して、数字に変換して戻す。
    //　4ビットｘ4行で16ビット。　パターンは6万通り？
    //■□□□□□
    //□ｘｘｘｘ□
    //□ｘｘｘｘ□
    //□ｘｘｘｘ□
    //□ｘｘｘｘ□
    //□□□□□□
    //
    int	whitecount=0;
    uchar* pointer;
    uchar* wkpointer;

    int cellRow=6;
    int cellCol=6;
    int result[4][4];
    int offset=5;
    int cellsize=10;
    int c_white=0;
    int cellx,celly;

    for (int i=0;i<4;i++)
    {
        for (int j=0;j<4;j++)
        {
            result[i][j]=0;
        }
    }

    //ドットを検出
    for(int y=0;y<4;y++)
    {
        pointer=(uchar*)(src->imageData + (y+2)*cellsize*src->widthStep);
        for(int x=0;x<4;x++)
        {
            c_white=0;
            if(pointer[src->nChannels*cellsize*(x+2)]>1)
            {

                for(celly=-2;celly<3;celly++)
                {
                    wkpointer=(uchar*)(src->imageData + (celly+((y+2)*cellsize))*src->widthStep);
                    for(cellx=-2;cellx<3;cellx++)
                    {
                        if(pointer[src->nChannels*(cellsize*(x+2)+cellx)]>1)
                        {
                            c_white++;
                        }
                    }

                }
                if(c_white>4)
                {
                    whitecount++;
                    result[y][3-x]=1;
                }
            }
        }
    }

    //向きを検出
    //

    //0°　　　(10,10)から上下左右3ピクセル
    c_white=0;
    for(celly=-3;celly<4;celly++)
    {
        wkpointer=(uchar*)(src->imageData + (celly+(10))*src->widthStep);
        for(cellx=-3;cellx<4;cellx++)
        {
            if(wkpointer[src->nChannels*(10+cellx)]>1)
            {
                c_white++;
            }
        }

    }
    if(c_white>30)
    {
        MarkerDirection=0;
    }

    //90deg　(60,10)の上下左右3ピクセル
    c_white=0;
    for(celly=-3;celly<4;celly++)
    {
        wkpointer=(uchar*)(src->imageData + (celly+(10))*src->widthStep);
        for(cellx=-3;cellx<4;cellx++)
        {
            if(wkpointer[src->nChannels*(60+cellx)]>1)
            {
                c_white++;
            }
        }

    }


    if(c_white>30)
    {
        MarkerDirection=90;
    }


    //180deg　(60,60)の上下左右3ピクセル
    c_white=0;
    for(celly=-3;celly<4;celly++)
    {
        wkpointer=(uchar*)(src->imageData + (celly+(60))*src->widthStep);
        for(cellx=-3;cellx<4;cellx++)
        {
            if(wkpointer[src->nChannels*(60+cellx)]>1)
            {
                c_white++;
            }
        }

    }
    if(c_white>30)
    {
        MarkerDirection=180;
    }

    //270deg　(10,60)の上下左右3ピクセル
    c_white=0;
    for(celly=-3;celly<4;celly++)
    {
        wkpointer=(uchar*)(src->imageData + (celly+(60))*src->widthStep);
        for(cellx=-3;cellx<4;cellx++)
        {
            if(wkpointer[src->nChannels*(10+cellx)]>1)
            {
                c_white++;
            }
        }
    }
    if(c_white>30)
    {
        MarkerDirection=270;
    }


    //これだとまだ画像に表示されたままの順番になっているので、
    //どちらを向いているかを検出して、その方向に配列を組みなおす必要がある。


    //
    int temp[4][4];
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            temp[i][j]=result[i][j];
        }
    }

    if(MarkerDirection==0)
    {
        //何もしない
    }
    if(MarkerDirection==180)
    {
        //全部反対にする
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<4;j++)
            {
                result[3-i][3-j]=temp[i][j];
            }
        }
    }

    if(MarkerDirection==90)
    {
        //
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<4;j++)
            {
                result[j][3-i]=temp[i][j];
            }
        }
    }

    if(MarkerDirection==270)
    {
        //
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<4;j++)
            {
                result[3-j][i]=temp[i][j];
            }
        }
    }


    whitecount=0;
    int nn=0;
    for (int i=0;i<4;i++)
    {
        for (int j=0;j<4;j++)
        {
            nn=	(int)pow((double)16,i)*result[i][j]*pow((double)2,j);
            whitecount=nn+ whitecount;;

        }
    }

    return whitecount;
}