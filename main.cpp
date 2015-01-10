#include <iostream>

#include "image_processing.h"
#include "control.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "include/cflie/CCrazyflie.h"




using namespace std;

int main(int argc, char **argv) {
    CCrazyRadio *crRadio = new CCrazyRadio("radio://0/10/250K");
    cv::VideoCapture cap;
    Drone *drone = new Drone();



    bool loop_flag = true;

    if (crRadio->startRadio()) {

        /*  init crazyflie */
        CCrazyflie *cflieCopter = new CCrazyflie(crRadio);
        cflieCopter->setThrust(0);
        cflieCopter->setSendSetpoints(true);

        imageInit();

        cv::namedWindow("output", 1);

        while (cflieCopter->cycle() && loop_flag) { //begin loop

            imageProcess(drone);
            control(drone,cflieCopter);
            int k = cvWaitKey(33);
            switch (k) {
                case 'q':
                case 'Q':
                    loop_flag = false;
                    break;
                case 's':
                    cflieCopter->setThrust(0);
                    cout << "set 0" << endl;
                    break;
                case 'a':
                    cflieCopter->setThrust(40001);
                    cout << "set 10001" << endl;
                    break;
                case 'z':
                    cflieCopter->setYaw(0.0);
                    cout << "set 0" << endl;
                    break;
                case 'x':
                    cflieCopter->setYaw(90.0);
                    cout << "set 10001" << endl;
                    break;
            }

        } //end loop

        delete cflieCopter;
    } else {
        cerr << "Could not connect to dongle. Did you plug it in?" << endl;
    }

    imageRelease();
    delete crRadio;
    return 0;
}