#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
from grasp_detection.srv import  graspRequest,graspRequestResponse
import DobotDllType as dType
import numpy as np
import json

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gxl/Downloads/Dobot-Magician-ROS-Demo/dobot_ws/src/dobot/src/DobotDll_x64

def load_transform(base_path, filename):
    s = None
    R = None
    T = None
    with open(base_path+ '/' + filename,'r') as f:
            p = json.load(f)
            s = np.array(p["s"])
            R = np.array(p["R"])
            T = np.array(p["T"])
    return s,R,T

def goto_point(api, dst, mode=dType.PTPMode.PTPMOVJXYZMode):
    dType.SetPTPCmd(api, mode, dst[0], dst[1], dst[2], dst[3], isQueued=1)

def grasp_client():
    rospy.init_node('grasp_client')
    rospy.wait_for_service("/grasp_request")
    try:
        grasp_client = rospy.ServiceProxy("/grasp_request",graspRequest)
        client_response = grasp_client(1)
        return client_response
    except rospy.ServiceException, e:
        print("Service call failed: %s"%e)

def check_angle(rev_angle):
    while rev_angle>63:
        rev_angle -= 180
    while rev_angle<-237:
        rev_angle += 180
    print("##rev_angle :",rev_angle)
    return rev_angle        

if __name__ == "__main__":
    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

    #Load Dll
    api = dType.load()

    #Connect Dobot
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    # 箱子坐标
    homex = 0
    homey = -200
    homez = 40
    homer = 0

    # 相机内参矩阵的逆矩阵,用于将图像坐标转换为相机坐标
    Minv = np.array([[0.0016221626347814378, 0.0, -0.5337791036253712], [0.0, 0.0016235716628296255, -0.39760458236866114], [0.0, 0.0, 1.0]])
    s,R,T = load_transform('/home/js/Documents/catkin_ws/src/grasp_detection/scripts', 'calibration_results.json')
    T = T.reshape(3,1)

    if (state == dType.DobotConnect.DobotConnect_NoError):
        #Clean Command Queued
        dType.SetQueuedCmdClear(api)
        
        dType.SetPTPJointParams(api,100,100,100,100,100,100,100,100)
        dType.SetPTPCoordinateParams(api,200,200,200,200)
        dType.SetPTPCommonParams(api, 100, 100)
        dType.SetPTPJumpParams(api, 20, 200)

        dType.ClearAllAlarmsState(api)

        dType.SetQueuedCmdStartExec(api)

        x3=homex;y3=homey;z3=homez;r3=0;#箱子上方
        x4=x3;y4=y3;z4=-28;r4=0;#箱子里

        #移动到home处,将机械臂移除视野
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, homex, homey, homez, homer, isQueued=1)
        dType.dSleep(ms=2000)

        grasp = grasp_client()
        print("cx = {} ; cy = {} ; angle = {} ; a.depth = {}\n".format(grasp.cx,grasp.cy,grasp.angle,grasp.depth))

        if grasp.depth == 0.0:
            grasp.depth = 460.

        #图像坐标系下坐标,齐次坐标
        image = np.zeros((3,1),dtype = np.float32)
        image[0,0] = grasp.cx
        image[1,0] = grasp.cy
        image[2,0] = 1.0

        camera = grasp.depth * np.dot(Minv,image)
        print("## Camera coordinate :",camera)

        target = s * np.dot(R,camera) + T
        print("##target :",target)

        #目标上方#待定 #a.angle+90 #-237~63
        p1 = target[0,0], target[1,0], 10, check_angle(grasp.angle)
        #目标处#待定
        p2 = p1[0], p1[1], -32, p1[3]
        p3 = homex, homey, homez, homer
        p4 = p3[0], p3[1], -28, p3[3]

        #移动到目标上方x1
        goto_point(api, p1)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=0, isQueued=1)#爪子张开#
        dType.SetWAITCmd(api, 1, isQueued=1)

        #移动到目标处
        #移动到x2处#
        goto_point(api, p2)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=1, isQueued=1)#爪子夹取#
        dType.SetWAITCmd(api, 2, isQueued=1)

        #移动到箱子上方
        #移动到x3处#
        goto_point(api, p3)

        #移动到箱子里x4处#
        goto_point(api, p4)
        #dType.dSleep(ms=2000)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=0, isQueued=1)#爪子张开#
        dType.SetWAITCmd(api, 2, isQueued=1)

        #移动到箱子上方#移动到x3处#
        goto_point(api, p3)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=1, isQueued=1)#爪子闭合#
        dType.SetWAITCmd(api, 1, isQueued=1)

        dType.SetEndEffectorGripper(api, enableCtrl=0,  on=1, isQueued=1)#关闭爪子

        #归零
        dType.SetHOMECmd(api,1,isQueued=1)   
    #Disconnect Dobot
    dType.DisconnectDobot(api)



    
    
