#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
from grasp_detection.srv import  graspRequest,graspRequestResponse
import DobotDllType as dType
import numpy as np
import json

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gxl/Downloads/Dobot-Magician-ROS-Demo/dobot_ws/src/dobot/src/DobotDll_x64

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
    if (state == dType.DobotConnect.DobotConnect_NoError):

        #Clean Command Queued
        dType.SetQueuedCmdClear(api)
        
        dType.SetPTPJointParams(api,100,100,100,100,100,100,100,100)
        dType.SetPTPCoordinateParams(api,200,200,200,200)
        dType.SetPTPCommonParams(api, 100, 100)
        dType.SetPTPJumpParams(api, 20, 200)

        dType.ClearAllAlarmsState(api)

        dType.SetQueuedCmdStartExec(api)

        #dType.SetHOMECmd(api, temp=1, isQueued=0)			#temp是无效值
        #dType.dSleep(ms=5000)							#延时5s操作

        homex = 0;homey = -200;homez = 40;homer = 0;
        xyzvel = 200;xyzacc = 200;rvel = 200;racc = 200;

        x3=homex;y3=homey;z3=homez;r3=0;#箱子上方
        x4=x3;y4=y3;z4=-28;r4=0;#箱子里

        #dType.SetHOMEParams(api,  homex,  homey,  homez,  homer,  isQueued=1)#设置回零参数#
        #dType.SetHOMECmd(api,1,1)

        #Start to Execute Command Queued

        #从家出发#移动到home处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, homex, homey, homez, homer, isQueued=1)
        dType.dSleep(ms=2000)

        a = grasp_client()
        print("cx = {} ; cy = {} ; angle = {} ; a.depth = {}\n".format(a.cx,a.cy,a.angle,a.depth))

        if a.depth == 0.0:
            a.depth = 460.

        T = None;s = None;R = None;
        Minv = np.array([[0.0016221626347814378, 0.0, -0.5337791036253712], [0.0, 0.0016235716628296255, -0.39760458236866114], [0.0, 0.0, 1.0]])
        tmp = np.zeros((3,1),dtype = np.float32);tmp[0,0] = a.cx;tmp[1,0] = a.cy;tmp[2,0] = 1.0;
        C = a.depth * np.dot(Minv,tmp)
        print("##c :",C)
        
        with open('/home/js/Documents/catkin_ws/src/grasp_detection/scripts/calibration_results.json','r') as f:
            p = json.load(f)
            #print(p)
            T = np.array(p["T"]);s = np.array(p["s"]);R = np.array(p["R"])
        print("##T = {} \nR = {} \ns = {} \n".format(T,R,s))
        T = T.reshape(3,1)
        print("##t.reshape :",T)

        TARGET = s * np.dot(R,C) + T
        print("##target :",TARGET)

        x1=TARGET[0,0];y1=TARGET[1,0];z1=10;r1=check_angle(a.angle);#目标上方#待定 #a.angle+90 #-237~63
        x2=x1;y2=y1;z2=-31;r2=r1;#目标处#待定 

        #移动到目标上方x1
        dType.SetPTPCoordinateParams(api, xyzvel, xyzacc, rvel,  racc,  isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x1, y1, z1, r1, isQueued=1)
        #dType.dSleep(ms=2000) 
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=0, isQueued=1)#爪子张开#
        dType.SetWAITCmd(api, 1, isQueued=1)

        #移动到目标处
        #移动到x2处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x2, y2, z2, r2, isQueued=1)
        #dType.dSleep(ms=2000) 
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=1, isQueued=1)#爪子夹取#
        dType.SetWAITCmd(api, 1, isQueued=1)
        #dType.dSleep(ms=2000) 

        #移动到目标上方x1
        #移动到x1处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x1, y1, z1, r1, isQueued=1)
        #dType.dSleep(ms=2000)

        #移动到箱子上方
        #移动到x3处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x3, y3, z3, r3, isQueued=1)
        #dType.dSleep(ms=2000)

        #移动到箱子里x4处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x4, y4, z4, r4, isQueued=1)
        #dType.dSleep(ms=2000)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=0, isQueued=1)#爪子张开#
        dType.SetWAITCmd(api, 1, isQueued=1)

        #移动到箱子上方#移动到x3处#
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x3, y3, z3, r3, isQueued=1)
        #dType.dSleep(ms=2000)
        dType.SetEndEffectorGripper(api, enableCtrl=1,  on=1, isQueued=1)#爪子闭合#
        dType.SetWAITCmd(api, 1, isQueued=1)

        dType.SetEndEffectorGripper(api, enableCtrl=0,  on=1, isQueued=1)#关闭爪子

        #归零
        dType.SetHOMECmd(api,1,isQueued=1)   

    #Clean Command Queued
    #dType.SetQueuedCmdClear(api)
    #Disconnect Dobot
    dType.DisconnectDobot(api)



    
    
