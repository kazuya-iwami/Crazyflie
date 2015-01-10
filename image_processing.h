#pragma once

#include <opencv2/opencv.hpp>
#include "control.h"


void imageInit();
void imageProcess(cv::Mat &input, Drone *drone);
void imageRelease();