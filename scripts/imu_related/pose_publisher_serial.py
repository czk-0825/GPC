#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import tf2_ros
import gatt
import sys
import numpy as np
import cv2
from scipy import io
from sensor_msgs.msg import Imu
import time
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped

def publish_imu(imu_pub,imu_data):
    imu = Imu()
    imu.header.frame_id = "world"
    imu.header.stamp = rospy.Time.now()
    imu.linear_acceleration.x = imu_data[0]
    imu.linear_acceleration.y = imu_data[1]
    imu.linear_acceleration.z = imu_data[2]
    imu.angular_velocity.x = imu_data[3]
    imu.angular_velocity.y = imu_data[4]
    imu.angular_velocity.z = imu_data[5]*np.pi/180
    imu.orientation.x = imu_data[9]
    imu.orientation.y = imu_data[10]
    imu.orientation.z = imu_data[11]
    imu.orientation.w = imu_data[12]
    imu_pub.publish(imu)
    return imu


if __name__ == "__main__":

    buffer_name = sys.argv[1]
    rospy.init_node("pose_publisher", anonymous=True)
    tf_imu_camera_pub = tf2_ros.TransformBroadcaster()
    tf_world_base_pub = tf2_ros.StaticTransformBroadcaster()

    imu_pub = rospy.Publisher(f"imu/{buffer_name}",Imu, queue_size = 10)
    rate = rospy.Rate(100)

    imu_head_bf = np.memmap(f"/home/czk/catkin_ws/src/samcon_perception/scripts/imu/log/imu_{buffer_name}.npy", dtype='float32', mode='r',shape=(13,))

    t0 = time.time()

    while not rospy.is_shutdown():
        try:
            t0 = time.time()
            imu_data = np.copy(imu_head_bf[0:])
            # imu_head = publish_imu(imu_head_pub,imu_data = imu_head_data)
            imu = Imu()
            imu.header.frame_id = "world"
            imu.linear_acceleration.x = imu_data[0]
            imu.linear_acceleration.y = imu_data[1]
            imu.linear_acceleration.z = imu_data[2]
            imu.angular_velocity.x = imu_data[3]
            imu.angular_velocity.y = imu_data[4]
            imu.angular_velocity.z = imu_data[5]*np.pi/180
            imu.orientation.x = imu_data[9]
            imu.orientation.y = imu_data[10]
            imu.orientation.z = imu_data[11]
            imu.orientation.w = imu_data[12]
            t = time.time()-t0
            imu.header.stamp = rospy.Time.now() - rospy.Duration(t)
            print(t)
            imu_pub.publish(imu)

        except Exception as e:
            rospy.logerr("Exception:%s", e)
            break
        time.sleep(0.005)
