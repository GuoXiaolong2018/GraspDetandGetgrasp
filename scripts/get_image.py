#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageManager:
    def __init__(self):
        self.rgd_topic = "/rgd_image"
        self.rgd_pub = rospy.Publisher(self.rgd_topic, Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/depth_processed", Image, queue_size=1)
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCallback)
        self.depth_topic = rospy.get_param("~depth_topic","/camera/aligned_depth_to_color/image_raw")
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depthCallback)
        self.rgb = None
        self.depth = None
        self.depthr_ = None # Re-ranged depth
        self.rgd = None
        self.cvb = CvBridge()
        rospy.spin()

    def imageCallback(self, data):
        self.rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.rgd = self.rgb.copy()
        if self.depthr_ is not None:
            self.rgd[:,:,2] = np.squeeze(self.depthr_)

            try:
                self.rgd_pub.publish(self.cvb.cv2_to_imgmsg(self.rgd, "rgb8"))
            except CvBridgeError as e:
                print(e)

    def depthCallback(self, data):
        self.depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width, -1)
        self.depthr_ = np.uint8((self.depth.astype(np.float32)-500.)/(1200.-500.)*255) # Re-range depth value to 0~255

        try:
            self.depth_pub.publish(self.cvb.cv2_to_imgmsg(self.depthr_, "mono8"))
        except CvBridgeError as e:
            print(e)
        

if __name__ == "__main__":
    rospy.init_node("get_image_node")
    rospy.loginfo("Start get_image_node")
    im = ImageManager()
