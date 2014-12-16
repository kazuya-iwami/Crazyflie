#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cflie/CCrazyflie.h>


using namespace std;

int main(int argc, char **argv) {
    CCrazyRadio *crRadio = new CCrazyRadio("radio://0/10/250K");
    cv::VideoCapture cap;
    cv::Mat frame;

    bool loop_flag = true;

    if (crRadio->startRadio()) {

        /*  init crazyflie */
        CCrazyflie *cflieCopter = new CCrazyflie(crRadio);
        cflieCopter->setThrust(0);
        cflieCopter->setSendSetpoints(true);

        /*  init camera */
        cap.open(0);
        if (!cap.isOpened()) {
            cout << "can't open camera" << endl;
            return 0;
        }

        cv::namedWindow("output", 1);

        /*  init backgroundSubstractor  */
        //cv::BackgroundSubtractorGMG backGroundSubtractor;




        while (cflieCopter->cycle() && loop_flag) { //begin loop
            //cout << cflieCopter->yaw() << " " << cflieCopter->roll()<< " " << cflieCopter->pitch();
            //cout << "thrust "<< cflieCopter->thrust() << endl;

            cap >> frame;
            if (frame.empty()) {
                break;
            }


            cv::imshow("output", frame);

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
            }

        } //end loop

        delete cflieCopter;
    } else {
        cerr << "Could not connect to dongle. Did you plug it in?" << endl;
    }

    delete crRadio;
    return 0;
}