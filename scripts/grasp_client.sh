#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/js/Documents/dobot_ws/src/dobot/src/DobotDll_x64
sudo chmod 777 /dev/ttyUSB*
/home/js/Documents/catkin_ws/src/grasp_detection/scripts/getGrasp.py
