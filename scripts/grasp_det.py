#!/usr/bin/env python
from grasp_detection.srv import graspRequest, graspRequestResponse

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
lib_path = '/home/js/Documents/catkin_ws/src/grasp_detection/scripts/lib'
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon

pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array

CLASSES = ('__background__',
           'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
           'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
           'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
           'angle_16', 'angle_17', 'angle_18', 'angle_19')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_240000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}

class DetectionManager:
    def __init__(self):
        self.ROOT_DIR = '/home/js/Documents/catkin_ws/src/grasp_detection'
        self.rgd_topic = "/rgd_image"
        self.rgd_pub = rospy.Publisher(self.rgd_topic, Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/depth_processed", Image, queue_size=10)
        self.result_pub = rospy.Publisher("/result_detection", Image, queue_size=10)
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        self.depth_topic = rospy.get_param("~depth_topic","/camera/aligned_depth_to_color/image_raw")
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depthCallback)
        self.detect_server = rospy.Service("grasp_request", graspRequest, self.handle_detection_request)
        self.grasp_center = [0., 0.]
        self.grasp_angle = 0.
        self.rgb = None
        self.depth = None
        self.depthr_ = None # Re-ranged depth
        self.rgd = None
        self.cvb = CvBridge()
        self.mask = cv2.imread(os.path.join(self.ROOT_DIR,'data/mask.jpg'),0)
        self.det_viz = None

        # Detection initialization
        self.root_path = '/home/js/Documents/catkin_ws/src/grasp_detection/scripts'
        self.demonet = 'res50'
        self.dataset = 'grasp'
        self.tfmodel = os.path.join(self.root_path,'output', self.demonet, DATASETS[self.dataset][0], 'default',
                              NETS[self.demonet][0])

        if not os.path.isfile(self.tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                        'our server and place them properly?').format(self.tfmodel + '.meta'))

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tfconfig)
        self.net = resnetv1(batch_size=1, num_layers=50)
        self.net.create_architecture(self.sess, "TEST", 20,
                          tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(self.sess, self.tfmodel)
        rospy.loginfo('Loaded network {:s}'.format(self.tfmodel))

        rospy.spin()

    def imageCallback(self, data):
        self.rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.rgd = self.rgb.copy()
        if self.depthr_ is not None:
            try:
                self.depth_pub.publish(self.cvb.cv2_to_imgmsg(self.depthr_, "mono8"))
            except CvBridgeError as e:
                print(e)

            self.rgd[:,:,2] = np.squeeze(self.depthr_)
            self.rgd = cv2.bitwise_and(self.rgd, self.rgd, mask=self.mask)

            try:
                self.rgd_pub.publish(self.cvb.cv2_to_imgmsg(self.rgd, "rgb8"))
            except CvBridgeError as e:
                print(e)

            if self.det_viz is not None:
                try:
                    self.result_pub.publish(self.cvb.cv2_to_imgmsg(self.det_viz, "rgb8"))
                except CvBridgeError as e:
                    print(e)

    def depthCallback(self, data):
        self.depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width, -1)
        temp = (self.depth.astype(np.float32)-440.)/(510.-440.) # Re-range depth value to 0~255
        temp[np.where(temp>1.0)] = 0.
        #temp[np.where(temp<0.0)] = 0.
        self.depthr_ = np.uint8(temp*255.)

        try:
            self.depth_pub.publish(self.cvb.cv2_to_imgmsg(self.depthr_, "mono8"))
        except CvBridgeError as e:
            print(e)

    def Rotate2D(self, pts,cnt,ang=scipy.pi/4):
        '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
        return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

    def detect(self, sess, net, im):
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)
        timer.toc()

        self.det_viz = self.rgb.copy()

        rospy.loginfo('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        # Visualize detections for each class
        CONF_THRESH = 0.1	
        NMS_THRESH = 0.3
        max_score = 0.0
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            self.vis_detections(im, cls, dets)
            if dets[0,4] > max_score:
                max_score = dets[0,4]
                self.grasp_center[0] = (dets[0,0] + dets[0,2])/2
                self.grasp_center[1] = (dets[0,1] + dets[0,3])/2
                tmp = int(cls[6:])
                self.grasp_angle = 9*(tmp-1)

    def vis_detections(self, im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            # plot rotated rectangles
            pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            angle = int(class_name[6:])
            r_bbox = self.Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
            r_bbox = np.array(r_bbox).astype(np.int32)
            #pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[0,0],r_bbox[0,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
            cv2.line(self.det_viz, (r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (0,0,0))
            cv2.line(self.det_viz, (r_bbox[1,0],r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (255,0,0))
            cv2.line(self.det_viz, (r_bbox[2,0],r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1]), (0,0,0))
            cv2.line(self.det_viz, (r_bbox[3,0],r_bbox[3,1]), (r_bbox[0,0], r_bbox[0,1]), (255,0,0))

    def vis_detection_max(self, im, class_name, dets):
        """Draw detected bounding boxes."""
        imOut = self.rgb.copy()
        ind = -1
        ind = np.argmax(dets[:, -1])
        if ind == -1:
            return

        bbox = dets[ind, :4]
        score = dets[ind, -1]

        # plot rotated rectangles
        pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        angle = int(class_name[6:])
        r_bbox = self.Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1))
        r_bbox = np.array(r_bbox).astype(np.int32)
        #pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[0,0],r_bbox[0,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
        cv2.line(imOut, (r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (0,0,0))
        cv2.line(imOut, (r_bbox[1,0],r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (255,0,0))
        cv2.line(imOut, (r_bbox[2,0],r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1]), (0,0,0))
        cv2.line(imOut, (r_bbox[3,0],r_bbox[3,1]), (r_bbox[0,0], r_bbox[0,1]), (255,0,0))

        self.result_pub.publish(self.cvb.cv2_to_imgmsg(imOut, "rgb8"))

    def handle_detection_request(self, req):
        rospy.loginfo("Request type is {}".format(req.request))
        if self.rgd is not None:
            self.detect(self.sess, self.net, self.rgd)
            rospy.loginfo("Grasp center:({},{})".format(self.grasp_center[0], self.grasp_center[1]))
            rospy.loginfo("Grasp angle:{}".format(self.grasp_angle))
            return graspRequestResponse(self.grasp_center[0], self.grasp_center[1], float(self.grasp_angle))
        else:
            rospy.logerr("RGD input image is NULL.")


if __name__ == "__main__":
    rospy.init_node("get_image_node")
    rospy.loginfo("Start get_image_node")
    im = DetectionManager()