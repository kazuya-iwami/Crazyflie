#pragma once
#include "include/cflie/CCrazyflie.h"

class Pos {
public:
    //Pos(int _x, int _y);
    int x;
    int y;
};

//Pos::Pos(int _x, int _y) {
//    this->x = _x;
//    this->y = _y;
//}

class Drone {
public:

    Pos cur_pos;//クアッドコプターの現在の画面上の位置
    Pos dst_pos;//目標座標

};

void control(Drone *drone,CCrazyflie *cflieCopter);



