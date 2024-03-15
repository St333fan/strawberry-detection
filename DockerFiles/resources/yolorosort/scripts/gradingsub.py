#!/usr/bin/env python3

# rospy for the subscriber

import rospy
# ROS Image message

from yolorosort.msg import fullBBox, singleBBox, cluster
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import docker_grading as gr

import numpy as np

from sklearn.cluster import DBSCAN
bridge = CvBridge()

class subscriber:

    def __init__(self):

        self.sub = rospy.Subscriber('/yolo/boundingboxes', fullBBox, self.callback)  # instantiate the Subscriber and Publisher
        #self.sub2 = rospy.Subscriber('/depth/image_raw', Image, self.callbackdepth)  # instantiate the Subscriber and Publisher



    def callback(self, data):
        for x in data.boxesWithAll:
            img = bridge.imgmsg_to_cv2(x.im, desired_encoding="8UC3")
            id = x.sortAlgoId
            depth = 200 #mm sollte aus callbackdepth commen
            if id > 99999:
                print("Ahhhh, there are too many strawberries!!")
                break
            else: 
                # change id into 5 digit number string
                id = "0"*(5-len(str(id)))+str(id)
    
                # evaluate picture quality
                first = gr.getrgbcontourimage(img.copy())
                sec1 = gr.bestpic(first, id)
                sec = sec1[0]

                # if its a better picture, tupel gets: original, firstcontour, convexhull, rotimgtolowest, initial grading  number
                if sec != "worse quality":
                    tupel = ((img, sec[0][0], sec[0][1], sec[0][2]), sec[2])
                    fingrade = gr.grade(tupel)
                    # folderpath where to save on harddrive, id, mass, grade, difference, jpg
                    name = str(id) + "_" + str(sec[1])[:10] + "_" + str(fingrade) + "_" + str(sec1[1]) + ".jpg"
                    print(name)
                    cv2.imwrite(os.path.join("/berry_pics",name), img)





def main():
    rospy.init_node('gradingnode', anonymous=True)
    obc = subscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
