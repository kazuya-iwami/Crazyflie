#pragma once

#include <opencv2/opencv.hpp>
#include "control.h"


void imageInit();
void imageProcess(Drone *drone);
void imageRelease();