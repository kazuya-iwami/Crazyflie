#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/core/core_c.h>
#include "control.h"
#include "image_processing.h"
#include <time.h>

using namespace std;
using namespace cv;

// base thrust
const int base_thrust=38000;
const int thrust_constant=2000;
const double lost_threshold=0.3;
// PID gain

// proportion
const double kp = 0.001;
//integrate
const double ki = 0.000;
//differentiate
const double kd = 0.000;

static double lost_t=0.0;
static double last_t = 0.0;//時間の保持(もとはint64とかいう形だった,,)
static double integral_x = 0.0, integral_y = 0.0;//積分用
static double previous_error_x = 0.0, previous_error_y = 0.0;//微分用


void control(Drone *drone,CCrazyflie *cflieCopter) {
    if(drone->captured_flag) {
        lost_t=0.0;
        //　目標座標までの差
        double error_x = drone->cur_pos.x - drone->dst_pos.x;
        double error_y = (drone->cur_pos.y - drone->dst_pos.y);//opencvの座標系ではyが大きくなるほど下に行くから

        // 時間 [s]
        double dt = (getTickCount() - last_t) / getTickFrequency();
        last_t = getTickCount();

        // 積分項

        if (dt > 0.1) {
            // リセット
            integral_x = 0.0;
            integral_y = 0.0;
        }
        integral_x += error_x * dt;
        integral_y += error_y * dt;

        // 微分項
        if (dt > 0.1) {
            // リセット
            previous_error_x = 0.0;
            previous_error_y = 0.0;
        }
        double derivative_x = (error_x - previous_error_x) / dt;
        double derivative_y = (error_y - previous_error_y) / dt;
        previous_error_x = error_x;
        previous_error_y = error_y;

        // 操作量
        double vx = kp * error_x + ki * integral_x + kd * derivative_x;
        double vy = kp * error_y + ki * integral_y + kd * derivative_y;
        double vz = 0.0;
        double vr = 0.0;
        std::cout << "(vx, vy)" << "(" << vx << "," << vy << ")" << "thrust" << base_thrust + thrust_constant * vy << std::endl;

        cflieCopter->setThrust(base_thrust + thrust_constant * vy);
    }else{//clock()は1秒程度の短時間用らしい
        double buf_t=(double)clock()/CLOCKS_PER_SEC-lost_t;
        //printf("%f\n",buf_t);
        if(lost_t==0.0){//lostした時の時間をメモ
            lost_t=(double)clock()/CLOCKS_PER_SEC;
            cflieCopter->setThrust(base_thrust);
        }
        else if(buf_t>lost_threshold){
            cflieCopter->setThrust(0);
            printf("totally lost. set thrust 0\n");
        }
    }
}