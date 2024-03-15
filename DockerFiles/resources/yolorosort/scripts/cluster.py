#!/usr/bin/env python3

# rospy for the subscriber

import rospy
# ROS Image message

from yolorosort.msg import fullBBox, singleBBox, cluster

import numpy as np

from sklearn.cluster import DBSCAN


class subscriber:

    def __init__(self):

        self.sub = rospy.Subscriber('/yolo/boundingboxes', fullBBox, self.callback)  # instantiate the Subscriber and Publisher
        self.pub = rospy.Publisher('/yolo/cluster_labels', cluster, queue_size=10)

    def callback(self, data):
        print("working on cluster... ")

        i = 0
        for sBox in data.boxesWithAll:
            if i == 0:
                X = np.array([[sBox.x, sBox.y]])
                i = 1
            else:
                X = np.append(X, [[sBox.x, sBox.y]], axis=0)

        if i > 0:
            print(X)
            clustering = DBSCAN(eps=0.2, min_samples=4).fit(X)
            print(clustering.labels_)

        clu = cluster()
        clu.header.seq = data.header.seq
        clu.header.stamp = rospy.get_rostime()

        clu.labels = clustering.labels_
        self.pub.publish(clu)


def main():
    rospy.init_node('cluster', anonymous=True)
    obc = subscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
