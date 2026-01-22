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

cam_to_imu = np.array([
    [ 0.99999235, -0.00228044, -0.00317894, 0.0],
    [0.00318958, 0.00467539, 0.99998398, 0.0],
    [-0.00226554, -0.99998647, 0.00468262, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
# cam_to_imu = np.array([
#     [ 0.99999235, 0.00228044, -0.00317894, 0.0],
#     [-0.00318958, 0.00467539, -0.99998398, 0.0],
#     [-0.00226554, 0.99998647, 0.00468262, 0.0],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# cam_to_imu = np.array([
#     [ 1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0],
#     [0.0, -1.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 1.0]
# ])

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

def publish_tf(tf_pub, imu_data):
    qx = imu_data[9]
    qy = imu_data[10]
    qz = imu_data[11]
    qw = imu_data[12]
    R_world_imu = R.from_quat([qx,qy,qz,qw]).as_matrix()
    R_imu_cam = R.from_euler('xyz',[0,0,90],degrees=True).as_matrix()
    R_world_cam = np.matmul(R_world_imu,R_imu_cam)
    tf_world_cam = TransformStamped()
    tf_world_cam.header.frame_id = "base"
    tf_world_cam.header.stamp = rospy.Time.now()
    tf_world_cam.child_frame_id = "camera_link"
    q_cam = R.from_matrix(R_world_cam).as_quat()
    tf_world_cam.transform.rotation.x = q_cam[0]
    tf_world_cam.transform.rotation.y = q_cam[1]
    tf_world_cam.transform.rotation.z = q_cam[2]
    tf_world_cam.transform.rotation.w = q_cam[3]
    tf_pub.sendTransform(tf_world_cam)

def publish_static_tf(tf_pub):
    tf_world_base = TransformStamped()
    tf_world_base.header.frame_id = "world"
    tf_world_base.header.stamp = rospy.Time.now()
    tf_world_base.child_frame_id = "base"
    tf_world_base.transform.translation.z = 1.5
    q_base = R.from_euler("xyz",[0,0,0]).as_quat()
    tf_world_base.transform.rotation.x = q_base[0]
    tf_world_base.transform.rotation.y = q_base[1]
    tf_world_base.transform.rotation.z = q_base[2]
    tf_world_base.transform.rotation.w = q_base[3]
    tf_pub.sendTransform(tf_world_base)

def publish_pose(pose_pub, imu_data, T_camera_link_to_optical):
    """
    计算 camera_color_optical_frame 在 world 坐标系下的姿态，并发布到指定的话题上。
    
    :param pose_pub: ROS 发布器 (geometry_msgs/PoseStamped)
    :param imu_data: 包含 IMU 数据的数组，索引 9:12 是四元数 (qx, qy, qz, qw)
    :param T_camera_link_to_optical: camera_link 到 camera_color_optical_frame 的变换矩阵 (4x4)
    """

    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [0, 0, 1.5]
    R_world_base = R.from_euler('xyz', [0, 0, 0]).as_matrix()
    T_world_base[:3, :3] = R_world_base

    qx, qy, qz, qw = -imu_data[9:13]
    R_world_imu = R.from_quat([qx, qy, qz, qw]).as_matrix()
    R_imu_cam = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    R_world_cam = np.matmul(R_world_imu, R_imu_cam)
    T_base_camera_link = np.eye(4)
    T_base_camera_link[:3, :3] = R_world_cam

    T_camera_link_to_optical = np.array(T_camera_link_to_optical)

    T_world_camera_optical = np.matmul(np.matmul(T_world_base, T_base_camera_link), T_camera_link_to_optical)

    translation = T_world_camera_optical[:3, 3]
    rotation_matrix = T_world_camera_optical[:3, :3]
    rotation = R.from_matrix(rotation_matrix).as_quat()

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "world"
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.pose.position.x = translation[0]
    pose_msg.pose.position.y = translation[1]
    pose_msg.pose.position.z = translation[2]
    pose_msg.pose.orientation.x = rotation[0]
    pose_msg.pose.orientation.y = rotation[1]
    pose_msg.pose.orientation.z = rotation[2]
    pose_msg.pose.orientation.w = rotation[3]

    # Step 7: 发布消息
    pose_pub.publish(pose_msg)

if __name__ == "__main__":

    buffer_name = sys.argv[1]
    rospy.init_node("pose_publisher", anonymous=True)
    tf_imu_camera_pub = tf2_ros.TransformBroadcaster()
    tf_world_base_pub = tf2_ros.StaticTransformBroadcaster()

    imu_head_pub = rospy.Publisher(f"{buffer_name}",Imu, queue_size = 10)
    pose_pub = rospy.Publisher("imu_color_to_world_pub",PoseStamped, queue_size = 10)
    rate = rospy.Rate(100)

    imu_head_bf = np.memmap(f"/home/czk/catkin_ws/src/samcon_perception/scripts/imu/{buffer_name}.npy", dtype='float32', mode='r',shape=(13,))
    
    
    img = np.zeros((300, 300), np.uint8)

    img.fill(200)

    t0 = time.time()

    while not rospy.is_shutdown():
        try:
            imu_head_data = np.copy(imu_head_bf[0:])
            imu_head = publish_imu(imu_head_pub,imu_data = imu_head_data)
            publish_pose(pose_pub, imu_head_data, cam_to_imu)
            publish_static_tf(tf_world_base_pub)
            publish_tf(tf_imu_camera_pub,imu_head_data)
            rospy.loginfo("Head:x=%.4f, y=%.4f, z=%.4f",
                        imu_head_bf[6], imu_head_bf[7], imu_head_bf[8])

            # cv2.imshow("Press q to stop imu", img)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        except Exception as e:
            rospy.logerr("Exception:%s", e)
            break
        time.sleep(0.008)
