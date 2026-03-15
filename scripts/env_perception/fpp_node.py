#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import time
from scipy.spatial.transform import Rotation as R
import threading
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from samcon_perception.msg import Glass_Gaze
import message_filters
import cv2
from queue import Queue
from matplotlib import colors as cl
from collections import defaultdict, namedtuple
from samcon_perception.msg import Gait_Info
import yaml
from math import cos, sin, radians

import open3d as o3d
import os
import copy
import sys
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from factor_utils import *
from gaze_pros_control.factor_utils import * #Standard ROS Package framework

YELLOW = "\033[93m"
RESET = "\033[0m"

class Feature():
    def __init__(self):

        self.feature_id = None

        self.first_ob_time = None

        self.foothold_type = []

        self.map_value = None

        self.center_position = []

        self.observation = dict()

class CAMState():
    def __init__(self):

        # An unique identifier for the CAM state.
        self.phase = None

        self.id = 0  # 0+

        self.step_num = 0

        self.timestamp = None

        self.img_rgb = []

        self.img_depth = []

        self.marker_pixel = None

        self.gaze_vector_in_world = None

        self.gaze_point = []

        self.gaze_in_world = []

        self.pcd_in_world = []

        self.ground_cloud = None

        self.transform_matrix = np.identity(4)

        self.imu_pose = np.identity(4)

        self.mask = None

        self.masked_rgb = None

class KeyFrame():
    def __init__(self):

        self.id = 0

        self.include_feature_id_list = []

        self.cam_state = CAMState()

        self.pcd_calib_no_ground = None

        self.calibr_matrix = np.identity(4)

class GaitState():
    def __init__(self):

        self.side = None

        self.step_length = None
        self.step_height = None
        self.foothold_type = None

        self.stance_t = None
        self.swing_t = None

        self.process_t = None

    def to_dict(self):
        if self.swing_t is None:
            self.swing_t = 0
        return {
            "side": self.side,
            "foothold_type": self.foothold_type[0],
            "step_length": round(float(self.step_length),4),
            "step_height": round(float(self.step_height),4),
            "foothold_slope":round(float(self.foothold_type[1]),4),
            "stance_time": round(float(self.stance_t),4),
            "swing_time": round(float(self.swing_t),4),
            "process_time": round(float(self.process_t),4)
        }

class Fusion():
    def __init__(self,maker = False):

        self.maker = maker

        self.cam_server = []
        self.cam_server_temp = []
        self.gait_server = []
        self.key_frame_server = dict()
        self.feature_server = dict()
        self.key_frame_id_list = []
        self.gaze_buf = []

        self.queue_size = 5
        self.down_sample_rate = 50 #降采样
        self.same_feature_threshold = 0.25 #同特征
        self.ground_threshold = 0.020 #去除地面
        self.min_dis = 0.09 #聚类
        self.min_num = 35 #聚类
        self.foothold_plane_threshold = 3
        if self.maker:
            self.queue_size = 5
            self.down_sample_rate = 20 #降采样
            self.same_feature_threshold = 0.25 #同特征
            self.ground_threshold = 0.02 #去除地面
            self.min_dis = 0.06 #聚类
            self.min_num = 50 #聚类
            self.foothold_plane_threshold = 3

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('gaze_pros_control')
        # self.base_dir = os.path.join(pkg_path, 'scripts/env_perception')
        self.sub = ''

        self.start_time = None
        self.init_pose = None
        self.init_ground = None
        self.next_frame_num = 0
        self.next_feature_id = 0
        self.is_feature_init = False
        self.nearest_feature_idx = None
        
        self.side = 'right'
        self.is_swing = False
        self.step_num = 0

        self.l_fpp_features = []
        self.r_fpp_features = []

        self._init_topics()

        self.bridge = CvBridge()
        self.open3d = Open3d_utils()
        self.gaze = Gaze_utils()
        self.gait_queue = Queue()
        self.condition = threading.Condition()

        self.show_points = None
        self.sphere_point = []

        self.gpp_thread = threading.Thread(target=self.processing_multi_clouds,daemon=True).start()
        self.pause_callback = threading.Event()

        self.r_stance_t = 0
        self.r_swing_t = 0
        self.l_stance_t = 0
        self.l_swing_t = 0
        self.swing_time = 0
        self.stance_time = 0

        self.saved_trans_mat = []
        self.saved_gaze_w = []
        self.saved_gaze_mat = []
        self.masked_rate = []
        self.rate = 0
        self.acc = 0

    def _init_topics(self):
        event_msg = rospy.Subscriber("/imu/gait_event", Gait_Info,self.event_callback)

        rgb_msg = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_msg = message_filters.Subscriber("/camera/depth/image_raw", Image)
        gaze_msg = message_filters.Subscriber("/glass/gaze", Glass_Gaze)
        pose_msg = message_filters.Subscriber("/camera/pose", PoseStamped)
        init_pose_msg = message_filters.Subscriber("/imu_color_to_world_pub", PoseStamped)
        mask_msg = message_filters.Subscriber("/camera/mask", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_msg, depth_msg, gaze_msg, pose_msg, init_pose_msg,mask_msg], self.queue_size, 0.5)
        self.ts.registerCallback(self.sync_pcd_pose)
    
    def add_new_features(self,segmented_terrain_pcd, segmented_labels, center_position,foothold_types,key_frame):
        new_labels = np.unique(segmented_labels)
        for cluster_id in range(len(new_labels)):
            new_feature = Feature()
            new_feature.feature_id = self.next_feature_id
            new_feature.first_ob_time = key_frame.cam_state.timestamp
            key_frame.include_feature_id_list.append(self.next_feature_id)
            new_feature.observation[key_frame.id] = segmented_terrain_pcd.select_by_index(np.where(segmented_labels == new_labels[cluster_id])[0])
            new_feature.center_position.append(center_position[cluster_id])
            new_feature.foothold_type.append(foothold_types[cluster_id])

            self.next_feature_id += 1
            # print(self.next_feature_id)
            self.feature_server[new_feature.feature_id] = new_feature

    def tracking_features(self,segmented_terrain_pcd, segmented_labels, center_position,foothold_types,key_frame,to_lib = True):
        tracked_labels = []
        tracked_centers =[]
        idxss = np.arange(len(np.unique(segmented_labels)))

        det_lists = []
        chosen_ids = []

        for f_id, old_feature in self.feature_server.items():

            for cluster_id,each_center_position in zip(np.unique(segmented_labels), center_position):
                dis = np.linalg.norm(each_center_position - np.mean(np.array(old_feature.center_position),axis = 0))
                if dis < self.same_feature_threshold:
                    
                    chosen_id = idxss[np.unique(segmented_labels) == cluster_id][0]
                    buf = [chosen_id,f_id,old_feature,cluster_id,each_center_position,dis]
                    if chosen_id in chosen_ids:
                        to_be_det = det_lists[np.arange(len(chosen_ids))[chosen_ids == chosen_id][0]]
                        if to_be_det[-1] >= dis:
                            det_lists[np.arange(len(chosen_ids))[chosen_ids == chosen_id][0]] = buf
                    else:
                        chosen_ids.append(chosen_id)
                        det_lists.append(buf)

        for det in det_lists:
            [chosen_id,f_id,old_feature,cluster_id,each_center_position] = det[:-1]
            key_frame.include_feature_id_list.append(f_id)
            if to_lib:
                old_feature.observation[key_frame.id] = segmented_terrain_pcd.select_by_index(
                                                np.where(segmented_labels == cluster_id)[0])

                self.feature_server[f_id].center_position.append(each_center_position)
                self.feature_server[f_id].foothold_type.append(foothold_types[chosen_id])
            tracked_labels.append(cluster_id)
            tracked_centers.append(each_center_position)

        mask = ~np.isin(segmented_labels, tracked_labels)
        idx = np.nonzero(mask)[0]
        segmented_labels = segmented_labels[mask]

        idxs = idxss[~np.isin(center_position, tracked_centers)[:,0]]
        foothold_types = [foothold_types[i] for i in idxs]
        center_position = center_position[idxs]

        segmented_terrain_pcd = segmented_terrain_pcd.select_by_index(idx)
        return segmented_terrain_pcd, segmented_labels, center_position,foothold_types
    
    def add_new_key_frame(self,camstate = None):
        key_frame = KeyFrame()
        if camstate is None:
            key_frame.cam_state = self.cam_server[-1]
            self.key_frame_id_list.append(key_frame.cam_state.id)
        else:
            key_frame.cam_state = camstate
        
        key_frame.calibr_matrix, plane_model, ground_indices,d = self.open3d.calc_cloud_in_ground(key_frame.cam_state.pcd_in_world,self.ground_threshold)
        normal = plane_model[:-1]
        vertical = np.array([0, 0, 1])
        self.foothold_plane_threshold = np.degrees(np.arccos(np.dot(normal, vertical) / (np.linalg.norm(normal) * np.linalg.norm(vertical))))

        if self.init_ground is None:
            self.init_ground = -d

        original_pcd = copy.copy(key_frame.cam_state.pcd_in_world).transform(key_frame.calibr_matrix)
        key_frame.pcd_calib_no_ground, _, key_frame.cam_state.ground_cloud,_ = self.open3d.remove_cloud_background(original_pcd, plane_model, ground_indices,remove_ground=True)
        key_frame.cam_state.ground_cloud = key_frame.cam_state.ground_cloud.transform(np.linalg.inv(key_frame.calibr_matrix))
        segmented_terrain_pcd, segmented_labels, foothold_slope,center_position = self.open3d.label_connected_area(
            key_frame,self.min_dis,self.min_num)
        if len(segmented_labels) == 0:
            key_frame.id = self.key_frame_id_list[-1]
            self.key_frame_server[key_frame.id] = key_frame
            return key_frame,[],[],[],[]

        foothold_types = self.foothold_type(foothold_slope)

        key_frame.id = key_frame.cam_state.id
        self.key_frame_server[key_frame.id] = key_frame
        return key_frame, segmented_terrain_pcd, segmented_labels, center_position,foothold_types
    
    def foothold_type(self,foothold_slope):
        foothold_types = []
        # print('foothold_plane_threshold:',self.foothold_plane_threshold)
        for i, slope in enumerate(foothold_slope):
            if abs(slope) < self.foothold_plane_threshold+3:
                slope_type = "levelground"
            elif slope >= self.foothold_plane_threshold+3:
                slope_type = "up_ramp"
            else:  # slope <= -3.0
                slope_type = "down_ramp" 

            foothold_types.append([slope_type, slope])

        return foothold_types

    def calc_nearest_fea(self,key_frame):
        center_dists = []
        center_dists_idx = []

        for f_id, old_feature in self.feature_server.items():
            center_dists_idx.append(f_id)
            center_position = np.mean(np.array(old_feature.center_position),axis=0)
            # distance = np.linalg.norm(center_position - key_frame.cam_state.gaze_in_world) # 距离注视点
            distance = np.linalg.norm(np.cross(center_position - key_frame.cam_state.gaze_vector_in_world[0], 
                                               center_position - key_frame.cam_state.gaze_vector_in_world[1]), axis=-1)#距离视线
            distance /= np.linalg.norm(key_frame.cam_state.gaze_vector_in_world[0]-key_frame.cam_state.gaze_vector_in_world[1])
            center_dists.append(distance)
        # print('min_distance:',center_dists[np.argmin(center_dists)])
        nearest_feature_idx = center_dists_idx[np.argmin(center_dists)]
        
        return nearest_feature_idx

    def fuse_nearest_fea(self,nearest_idx):
        open_pcd = None
        fuse_terrians = None
        time_factor = None
        first_ob_time = self.feature_server[nearest_idx].first_ob_time
        foothold_type = self.feature_server[nearest_idx].foothold_type

        for k_id,ob in self.feature_server[nearest_idx].observation.items():
            if fuse_terrians is None:
                open_pcd = ob
                fuse_terrians = np.array(ob.points)
                time_factor= np.full(len(ob.points), self.key_frame_server[k_id].cam_state.timestamp - first_ob_time)
            else:
                open_pcd += ob
                fuse_terrians = np.vstack((fuse_terrians,np.array(ob.points))).reshape(-1,3) 
                time_factor = np.concatenate((time_factor,np.full(len(ob.points), self.key_frame_server[k_id].cam_state.timestamp - first_ob_time)))

        dist_A = fuse_terrians[:,2] - np.mean(fuse_terrians[:,2])
        chosen_idx = dist_A < np.mean(dist_A) + 1 * np.std(dist_A)
        open_pcd = open_pcd.select_by_index(np.where(chosen_idx)[0])
        fuse_terrians = fuse_terrians[chosen_idx]
        time_factor = time_factor[chosen_idx]
        time_window_size = np.unique(time_factor)
        time_window_size = time_window_size[np.argmax(np.unique(time_factor))]
        return fuse_terrians,time_factor,time_window_size,open_pcd,foothold_type
    
    def calc_gait_info(self,key_frame,prev_key_frame,prev_fpp_feature):

        self.nearest_feature_idx = self.calc_nearest_fea(key_frame)
        # print(f'nearest_idx:{nearest_feature_idx}')
        pcd, time_factors, time_window_size, open_pcd ,foothold_type= self.fuse_nearest_fea(self.nearest_feature_idx)

        gait = np.zeros(2)
        gait1 = np.zeros(2)
        distances = np.linalg.norm(np.cross(pcd - key_frame.cam_state.gaze_vector_in_world[0], 
                                                   pcd - key_frame.cam_state.gaze_vector_in_world[1]), axis=-1)
        distances /= np.linalg.norm(key_frame.cam_state.gaze_vector_in_world[0]-key_frame.cam_state.gaze_vector_in_world[1])

        tt=time.time()
        if np.any(distances < 3e-2):
            current_fpp_point,map_value = self.gaze.time_attention_map(pcd,distances,time_factors,time_window_size) # 2
            # self.sphere_point.append(current_fpp_point)
            # current_fpp_point = key_frame.cam_state.gaze_in_world # 直接法
            # current_fpp_point = pcd[np.argmin(distances)] # 最近点法
        else:
            print("There are not distances greater than 3cm.")
            current_fpp_point,map_value = self.gaze.time_attention_map(pcd,distances,time_factors,time_window_size) # 2
            # self.sphere_point.append(current_fpp_point)

        if prev_fpp_feature is None:
            prev_fpp_point = np.array([0,0,self.init_ground])
        else:
            prev_fpp_point = prev_fpp_feature.foothold_in_world

        gait[0] = np.linalg.norm(current_fpp_point[:2]-prev_fpp_point[:2])
        gait[1] = current_fpp_point[2] - prev_fpp_point[2]

        if prev_key_frame is None:
            print(gait) 

        else:
            prev_fpp_point_base = np.matmul(prev_fpp_point - np.array([prev_key_frame.cam_state.transform_matrix[0,3],
                                                                       prev_key_frame.cam_state.transform_matrix[1,3],
                                                                       prev_key_frame.cam_state.transform_matrix[2,3]]),
                                            prev_key_frame.cam_state.transform_matrix[:3,:3])
                                        
            current_fpp_point_base = np.matmul(current_fpp_point - np.array([prev_key_frame.cam_state.transform_matrix[0,3],
                                                                       prev_key_frame.cam_state.transform_matrix[1,3],
                                                                       prev_key_frame.cam_state.transform_matrix[2,3]]),
                                            prev_key_frame.cam_state.transform_matrix[:3,:3])
            
            prev_fpp_point_base = np.matmul(prev_fpp_point_base,prev_key_frame.cam_state.imu_pose[:3,:3].T)
            current_fpp_point_base = np.matmul(current_fpp_point_base,prev_key_frame.cam_state.imu_pose[:3,:3].T)

            gait[0] = np.linalg.norm(current_fpp_point_base[:2]-prev_fpp_point_base[:2])
            gait[1] = current_fpp_point_base[2] - prev_fpp_point_base[2] 
            print('gait:',gait) 

        # print('max_map_value:',np.max(map_value))
        # print('min_map_value:',np.min(map_value))
        self.feature_server[self.nearest_feature_idx].map_value = map_value

        norm_map_value = (map_value - np.min(map_value)) / (np.max(map_value) - np.min(map_value))
        colors= plt.cm.jet(norm_map_value)[:, :-1]
        map_color_pcd = copy.copy(open_pcd)

        map_color_pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.show_points is None:
            self.show_points = map_color_pcd
        else:
            self.show_points += map_color_pcd

        slopes = []
        for foot in foothold_type:
            slopes.append(foot[1])
        slopes = np.array(slopes).reshape(-1)
        if len(slopes)>1:
            chosen_idx = slopes < np.mean(slopes) + 1 * np.std(slopes)
        else:
            chosen_idx = 0
        avg_slope = np.mean(slopes[chosen_idx])
        gait_feature = namedtuple('gait_feature', ['step_length', 
                                                   'step_height',
                                                   'foothold_type',
                                                   'foothold_in_world',
                                                   'absolute_gaze'])(gait[0],gait[1],self.foothold_type([avg_slope])[0],current_fpp_point,key_frame.cam_state.gaze_in_world)
        print(f'avg_type:{gait_feature.foothold_type}')
        return gait_feature

    def processing_multi_clouds(self, channel = None, data = None):
        while not rospy.is_shutdown():    
            with self.condition:
                self.condition.wait()
            # time.sleep(0.5)
            self.pause_callback.set()

            if self.side is not None:
                t = time.time()

                gs = GaitState()
                gs.side = self.side
                gs.stance_t = self.stance_time

                curr_fpp_feature = None
                prev_key_frame = None
                prev_fpp_feature = None

                if len(self.key_frame_server)>1:
                    prev_key_frame = self.key_frame_server[self.key_frame_id_list[-2]]

                    if self.side == 'left':
                        prev_fpp_feature = self.l_fpp_features[-1]
                    else:
                        prev_fpp_feature = self.r_fpp_features[-1]

                key_frame, segmented_terrain_pcd, segmented_labels, center_position, foothold_types = self.add_new_key_frame()
                if len(segmented_labels) != 0:
                    if self.is_feature_init:
                        segmented_terrain_pcd, segmented_labels, center_position, foothold_types = self.tracking_features(segmented_terrain_pcd, 
                                                                                    segmented_labels, center_position,foothold_types,key_frame,)

                    self.add_new_features(segmented_terrain_pcd, segmented_labels, center_position,foothold_types,key_frame)

                curr_fpp_feature = self.calc_gait_info(key_frame,prev_key_frame,prev_fpp_feature)

                gs.step_length = curr_fpp_feature.step_length
                gs.step_height = curr_fpp_feature.step_height
                gs.foothold_type = curr_fpp_feature.foothold_type

                if self.side == 'left':
                    self.l_fpp_features.append(curr_fpp_feature)
                else:
                    self.r_fpp_features.append(curr_fpp_feature)

                if not self.is_feature_init:
                    self.is_feature_init = True

                gs.process_t = time.time()-t
                print(f'processing time3: {time.time()-t}\r\n')

                self.gait_server.append(gs)
                self.pause_callback.clear()
            else:
                print('Closing...')

    def sync_pcd_pose(self, rgb_msg, depth_msg, gaze_msg, pose_msg, imu_msg,mask_msg):

        t = time.time()
        if self.start_time is None:
            self.start_time = t

        # if self.pause_callback.is_set():
        #     return

        if self.init_pose is None:
            self.init_pose = np.identity(4)
            init_rot = [imu_msg.pose.orientation.x,imu_msg.pose.orientation.y,imu_msg.pose.orientation.z,imu_msg.pose.orientation.w]
            euler = R.from_quat(init_rot).as_euler('xyz', degrees=True)
            if not self.maker:
                self.init_pose[:3,:3] = R.from_euler('xyz', [euler[0], euler[1], 0], degrees=True).as_matrix()
            else:
                self.init_pose[:3,:3] = R.from_euler('xyz', [euler[0], euler[1], euler[2]], degrees=True).as_matrix()

        cam = CAMState()
        cam.id = self.next_frame_num
        cam.timestamp = time.time() - self.start_time #rgb_msg.header.stamp.to_sec()
        # print(cam.timestamp)
        cam.img_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cam.img_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cam.masked_rgb = cam.img_rgb.copy()

        red_layer = np.zeros_like(cam.masked_rgb)
        red_layer[:] = (0, 0, 255)

        red_overlay = cv2.bitwise_and(red_layer, red_layer, mask=mask)
        alpha = 0.5
        overlayed = cv2.addWeighted(cam.masked_rgb, 1 - alpha, red_overlay, alpha, 0)
        cam.masked_rgb[mask.astype(bool)] = overlayed[mask.astype(bool)]

        if np.any(mask):
            cam.mask = mask
            self.rate = np.sum(mask == 255)/mask.size
        else:
            # if self.rate != 0:
            #     cv2.imwrite(f'1/{self.next_frame_num}_pre.png', self.cam_server[-1].masked_rgb)
            #     cv2.imwrite(f'1/{self.next_frame_num}_pre_mask.png', self.cam_server[-1].img_depth_masked)
            #     cv2.imwrite(f'1/{self.next_frame_num}_cur.png', cam.img_rgb)
            #     cv2.imwrite(f'1/{self.next_frame_num}_cur_depth.png', cam.img_depth)
            self.rate = 0
        self.masked_rate.append(self.rate)

        imu_rot = [imu_msg.pose.orientation.x,imu_msg.pose.orientation.y,imu_msg.pose.orientation.z,imu_msg.pose.orientation.w]
        euler = R.from_quat(imu_rot).as_euler('xyz', degrees=True)
        if not self.maker:
            cam.imu_pose[:3,:3] = R.from_euler('xyz', [euler[0], euler[1], 0], degrees=True).as_matrix()
        else:
            cam.imu_pose[:3,:3] = R.from_euler('xyz', [euler[0], euler[1], euler[2]], degrees=True).as_matrix()

        rot = [pose_msg.pose.orientation.x,pose_msg.pose.orientation.y,pose_msg.pose.orientation.z,pose_msg.pose.orientation.w]
        transform_mat = np.identity(4)
        transform_mat[0,-1] = pose_msg.pose.position.x
        transform_mat[1,-1] = pose_msg.pose.position.y
        transform_mat[2,-1] = pose_msg.pose.position.z
        rot_mat = R.from_quat(rot).as_matrix()
        transform_mat[:3, :3] = rot_mat

        cam.transform_matrix = np.dot(self.init_pose,transform_mat)

        u_2d = gaze_msg.u_2d
        v_2d = gaze_msg.v_2d
        # print(self.is_swing)

        if self.is_swing:
            self.gaze_buf.append([v_2d, u_2d])
            if len(self.gaze_buf)>5:
                self.gaze_buf = self.gaze_buf[1:]
            uv_tracker = np.mean(self.gaze_buf,axis = 0)
        else:
            self.gaze_buf = []
        uv_tracker = np.array([v_2d, u_2d])

        cam.gaze_point,cloud,gaze_vector,gaze_in_camera = self.gaze.tracker_to_rgbd_without_depth(uv_tracker, cam, self.down_sample_rate,mask)
        
        cam.gaze_vector_in_world = np.matmul(gaze_vector,cam.transform_matrix[:3,:3].T)+\
                                        np.array([cam.transform_matrix[0,3],cam.transform_matrix[1,3],cam.transform_matrix[2,3]])
        cam.gaze_in_world = np.matmul(gaze_in_camera,cam.transform_matrix[:3,:3].T)+\
                                        np.array([cam.transform_matrix[0,3],cam.transform_matrix[1,3],cam.transform_matrix[2,3]])
        cam.pcd_in_world = cloud.transform(cam.transform_matrix)
        self.saved_trans_mat.append(cam.transform_matrix)
        self.saved_gaze_w.append(cam.gaze_in_world)

        if self.maker:
            points_in_world = np.array(copy.copy(cam.pcd_in_world.points))
            # print("点云Z轴前10个值:", points_in_world[:100, 2])
            condition = (np.abs(points_in_world[:,0]) < 0.5) & (np.abs(points_in_world[:,2]+1.5) < 0.2) & (np.abs(points_in_world[:,1]) < 2.5)
            chosen_idx = np.arange(len(points_in_world))[np.where(condition)[0]]
            cam.pcd_in_world = cam.pcd_in_world.select_by_index(chosen_idx)
        # if self.is_swing:
        cam.step_num = self.step_num

        self.cam_server.append(cam)
        # if len(self.cam_server) > 15 and not self.maker:
        #     self.cam_server = self.cam_server[1:]

        self.next_frame_num += 1   
        if self.next_frame_num == 1:
            self.sub = 'xhy_6'
            # np.save(self.base_dir + f'/CZK/{self.sub}_rot.npy',self.init_pose)
            # cv2.imwrite(self.base_dir + f'/CZK/{self.sub}_rgb.png', cam.img_rgb)
            # cv2.imwrite(self.base_dir + f'/CZK/{self.sub}_depth.png', cam.img_depth)
        # print(time.time() - t)

    def post_processing(self):
        swing_cams = []
        phases = []
        for i in range(1,8):
            cams = []
            k_id = self.key_frame_id_list[i-1]
            for cam in self.cam_server:
                # if cam.step_num == i and cam.id <= k_id:
                if cam.step_num == i:
                    img = copy.copy(cam.masked_rgb)
                    cv2.circle(img, (cam.gaze_point[0], cam.gaze_point[1]), 15, (0, 0, 255), 6)
                    # cv2.imwrite(self.base_dir+f'/CZK/img/ori_rgb_step{i}_{cam.id}.png', cam.img_rgb)
                    # cv2.imwrite(self.base_dir+f'/CZK/img/gaze_rgb_step{i}_{cam.id}.png', img)
                    cams.append(cam)
                # if i == 5 and cam.id == 75 and len(cams)!=0:
                #     img = copy.copy(cam.masked_rgb)
                #     cv2.circle(img, (cam.gaze_point[0], cam.gaze_point[1]), 15, (0, 0, 255), 6)
                #     # cv2.imwrite(self.base_dir+f'/CZK/img3/ori_rgb_step{i}_{cam.id}.png', cam.img_rgb)
                #     # cv2.imwrite(self.base_dir+f'/CZK/img3/gaze_rgb_step{i}_{cam.id}.png', img)
                #     self.show_fused_pcd(cam,cams)
            if len(cams) != 0:
                swing_cams.append(cams)

        self.fuse_video(swing_cams,self.key_frame_id_list)

            # for cam in cams:
            #     img = copy.copy(cam.masked_rgb)
            #     cv2.circle(img, (cam.gaze_point[0], cam.gaze_point[1]), 15, (0, 0, 255), 6)
            #     cv2.imwrite(self.base_dir+f'/CZK/img/ori_rgb_step{i}_{cam.id}.png', cam.img_rgb)
            #     cv2.imwrite(self.base_dir+f'/CZK/img/gaze_rgb_step{i}_{cam.id}.png', img)
        
        print('step_num:',len(swing_cams))
        for i in range(len(swing_cams)):
            phase = []
            base_time = swing_cams[i][0].timestamp
            time_win_len = swing_cams[i][-1].timestamp - swing_cams[i][0].timestamp
            print(f'step_{i+1}: {len(swing_cams[i])} cam_states with {1000*time_win_len} ms')

            for cam in swing_cams[i]:
                cam.phase = int(100*(cam.timestamp - base_time)/time_win_len)
                phase.append(cam.phase)
            phases.append(phase)
            print(phase)
        shortest_row = min(phases, key=len)
        print(shortest_row)
        swing_cams_temp = []
        for i,phase in enumerate(phases):
            arr1 = np.array(phase)
            arr2 = np.array(shortest_row)
            diffs = np.abs(arr1[:, np.newaxis] - arr2)
            closest_indices = np.argmin(diffs, axis=0)
            selected_cams = [swing_cams[i][int(idx)] for idx in closest_indices]  # 显式转为 int
            swing_cams_temp.append(selected_cams)
        swing_cams_temp = np.array(swing_cams_temp, dtype=object)  # dtype=object 允许混合类型
        # print("NumPy 数组形状:", arr.shape)
        # self.data_replay(swing_cams_temp.T,shortest_row)

    def find_real_fpp(self,end_cam,is_masked = 1):
        cams_temp = []
        for cam in self.cam_server_temp:
            if cam.id < end_cam.id:
                cams_temp.append(cam)

        if is_masked:
            for each_cam in reversed(cams_temp):
                if each_cam.mask is not None:
                    v, u = np.where(each_cam.mask == 255)
                    u = np.min(u)
                    v = v[np.argmin(u)]
                    cam = each_cam
                    break
        else:
            cam = end_cam
            cam = cams_temp[-2]
            if self.side == 'right':
                u = 320
                v = 450
            else:
                u = 320
                v = 450
        fdw = self.gaze.get_point_in_world(int(u),int(v),cam)
        self.sphere_point.append(fdw)
        diss = []
        id = []
        for f_id, old_feature in self.feature_server.items():
            dis = np.linalg.norm(fdw[:2] - np.mean(np.array(old_feature.center_position),axis=0)[:2])
            diss.append(dis)
            id.append(f_id)
        real_id = id[np.argmin(diss)]
        if real_id == self.nearest_feature_idx:
            self.acc += 1

    def data_replay(self,cam_pair,phases):

        self.sub ='czk_1'
        is_right_first = True
        pre_gaze = []
        ab_gaze = []
        gait_params = []

        for i,cams in enumerate(cam_pair):
            gaze = []
            ab = []
            foothold_type = []
            ground = []
            gait_param = []

            self.key_frame_server.clear()  # 清空现有字典
            self.feature_server.clear()
            self.l_fpp_features = []
            self.r_fpp_features = []
            self.init_ground = None
            self.next_feature_id = 0
            self.is_feature_init = False
            self.cam_server = []
            self.key_frame_id_list = []
            self.show_points = None
            self.sphere_point = []
            self.acc = 0
            self.nearest_feature_idx = 0
            # if i > 10:
            if True:
                if is_right_first:
                    self.side == 'left'
                for j,cam in enumerate(cams):
                    self.cam_server.append(cam)

                    if self.side == 'right':
                        self.side = 'left'
                    else:
                        self.side = 'right'

                    with self.condition:
                        self.condition.notify()
                    time.sleep(0.1)
                    self.find_real_fpp(cam_pair[-1][j],is_masked = 0)
                print(f'phase:{phases[i]} with acc {self.acc}')
                # cv2.namedWindow("RGB Image")
                # cv2.imshow("RGB Image with Depth", cams[0].img_rgb)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows() 
                # self.show_feature_by_feature()
                # print(len(self.l_fpp_features),len(self.r_fpp_features))
            for lf,rf in zip(self.l_fpp_features,self.r_fpp_features):

                if is_right_first:
                    ab.append(rf.absolute_gaze)
                    ab.append(lf.absolute_gaze)
                    gaze.append(rf.foothold_in_world)
                    gaze.append(lf.foothold_in_world)
                    gait_param.append([rf.step_length,rf.step_height])
                    gait_param.append([lf.step_length,lf.step_height])
                    foothold_type.append(rf.foothold_type)
                    foothold_type.append(lf.foothold_type)
                else:
                    ab.append(lf.absolute_gaze)
                    ab.append(rf.absolute_gaze)
                    gaze.append(lf.foothold_in_world)
                    gaze.append(rf.foothold_in_world)
                    gait_param.append([lf.step_length,lf.step_height])
                    gait_param.append([rf.step_length,rf.step_height])
                    foothold_type.append(lf.foothold_type)
                    foothold_type.append(rf.foothold_type)

            pre_gaze.append(gaze)
            ab_gaze.append(ab)
            ground.append(self.init_ground)
            gait_params.append(gait_param)
            
        # if self.maker:
            # np.save(self.base_dir + f'/CZK/{self.sub}_gaze_all_frame.npy',self.saved_gaze_w)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_ab_gaze.npy',ab_gaze)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_pre_gaze.npy',pre_gaze)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_foothold.npy',foothold_type)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_phase.npy',phases)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_ground.npy',ground)
        #     np.save(self.base_dir + f'/CZK/{self.sub}_gait_param.npy',gait_params)
        # else:
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_pre_gaze.npy',pre_gaze)
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_ab_gaze.npy',ab_gaze)
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_foothold.npy',foothold_type)
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_phase.npy',phases)
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_ground.npy',ground)
        #     np.save(self.base_dir + f'/CZK/field_{self.sub}_gait_param.npy',gait_params)

    def event_callback(self,event_msg):
        if event_msg.event == 'toe_off':
            with self.condition:
                self.condition.notify()
            self.is_swing = True
            self.step_num += 1
            if event_msg.side == 'right':
                self.r_stance_t = time.time()
                t = time.time() - self.r_swing_t
                if t > 1e3:
                    self.stance_time = 0
                else:
                    self.stance_time = t
                self.side = 'right'
            if event_msg.side == 'left':
                self.l_stance_t = time.time()
                t = time.time() - self.l_swing_t
                if t > 1e3:
                    self.stance_time = 0
                else:
                    self.stance_time = t
                self.side = 'left'
        if event_msg.event == 'heel_strike':
            # self.find_real_fpp()
            self.is_swing = False
            if event_msg.side == 'right':
                self.r_swing_t = time.time()
                swing_time = time.time() - self.r_stance_t
                for gs in self.gait_server:
                    if gs.swing_t is None and gs.side == 'right':
                        gs.swing_t = swing_time
            if event_msg.side == 'left':
                self.l_swing_t = time.time()
                swing_time = time.time() - self.l_stance_t
                for gs in self.gait_server:
                    if gs.swing_t is None and gs.side == 'left':
                        gs.swing_t = swing_time
                
    def display_image(self):
        if len(self.cam_server)>0:
            img = np.copy(self.cam_server[-1].masked_rgb)
            cv2.circle(img, (self.cam_server[-1].gaze_point[0], self.cam_server[-1].gaze_point[1]), 15, (0, 0, 255), 6)
            cv2.imshow("ORB-SLAM3 RGB Image", img)
            cv2.waitKey(5)
        else:
            print('No image received')
            # time.sleep(0.3)

    def show_feature_by_feature(self):

        color_map1 = plt.get_cmap('tab20')
        color_map2 = plt.get_cmap('tab20b')
        color_map3 = plt.get_cmap('tab20c')

        combined_colors = np.concatenate([
            color_map1(np.linspace(0, 1, 20)),
            color_map2(np.linspace(0, 1, 20)),
            color_map3(np.linspace(0, 1, 20))
        ])

        show_pcd = None
        for f_id, one_feature in self.feature_server.items():

            feature_color = combined_colors[f_id][:3]
            pcd = None

            for k_id, obs in one_feature.observation.items():
                # print(f_id)
                ob = copy.copy(obs)
                colors = np.tile(feature_color, (len(ob.colors), 1))
                ob.colors = o3d.utility.Vector3dVector(colors)
                if pcd is None:
                    pcd = ob
                else:
                    pcd += ob

            fuse_terrians = np.array(pcd.points)
            dist_A = fuse_terrians[:,2] - np.mean(fuse_terrians[:,2])
            pcd = pcd.select_by_index(np.arange(len(fuse_terrians))[dist_A < np.mean(dist_A) + 1 * np.std(dist_A)])

            if show_pcd is None:
                show_pcd = pcd
            else:
                show_pcd += pcd

        fused_pcd = None
        i=0
        for k_id,key_frame in self.key_frame_server.items():
            if fused_pcd is None:
                fused_pcd = key_frame.cam_state.pcd_in_world
            else:
                fused_pcd += key_frame.cam_state.pcd_in_world
        # show_pcd += fused_pcd

        self.open3d.view_pcd_with_axis(show_pcd)

        line_set = o3d.geometry.LineSet()
        line_points = []
        lines = []
        colors = []
        spheres = []

        for k_id in self.key_frame_id_list:
            key_frame = self.key_frame_server[k_id]
            # start_point = key_frame.cam_state.gaze_vector_in_world[0].reshape(-1)
            start_point = key_frame.cam_state.gaze_vector_in_world[1].reshape(-1)
            end_point = key_frame.cam_state.gaze_in_world

            z_to_gaze_mat = np.identity(4)  # 创建单位矩阵，用于变换
            z_vec = end_point - start_point  # 计算两点之间的方向向量
            z_to_gaze_mat[:3, 2] = z_vec / np.linalg.norm(z_vec)  # 归一化并设置为z方向
            z_to_gaze_mat[:3, 0] = np.cross(z_to_gaze_mat[:3, 1], z_to_gaze_mat[:3, 2])  # 计算x方向的向量
            z_to_gaze_mat[:3, 0] /= np.linalg.norm(z_to_gaze_mat[:3, 0])  # 归一化x方向向量
            z_to_gaze_mat[:3, 3] = (start_point + end_point) / 2  # 设置圆柱体的中心位置

            height = np.linalg.norm(start_point - end_point)  # 计算圆柱体的高度
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=height)  # 创建圆柱体网格
            mesh_cylinder.paint_uniform_color([1,0,0])  # 为圆柱体上色
            mesh_cylinder_origin = copy.deepcopy(mesh_cylinder)  # 复制原始圆柱体网格
            mesh_cylinder.transform(z_to_gaze_mat)  # 对圆柱体应用变换矩阵

            idx = len(line_points)
            line_points.append(start_point.tolist())
            line_points.append(end_point.tolist())
            lines.append([idx, idx + 1])
            colors.append([1, 0, 0])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(key_frame.cam_state.gaze_in_world)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            spheres.append(sphere)
            spheres.append(mesh_cylinder)

        for point in self.sphere_point:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(point)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])
            spheres.append(sphere)

        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.open3d.view_pcd_with_axis(self.show_points,line_set,spheres)

    def show_fused_pcd(self,cam_state,cams):
        _,cloud,_,_ = self.gaze.tracker_to_rgbd_without_depth(cam_state.gaze_point,cam_state,5,cam_state.mask)
        cloud = cloud.transform(cam_state.transform_matrix)
        cam_state.pcd_in_world = cloud
        self.min_dis = 0.05
        self.min_num = 100
        self.ground_threshold = 0.016
        key_frame, segmented_pcd, segmented_labels, center_position,foothold_types = self.add_new_key_frame(cam_state)
        segmented_terrain_pcd, segmented_labels, center_position, foothold_types = self.tracking_features(segmented_pcd, 
                                                                                    segmented_labels, center_position,foothold_types,key_frame,to_lib=False)
        self.add_new_features(segmented_terrain_pcd, segmented_labels, center_position,foothold_types,key_frame)

        print(key_frame.include_feature_id_list)
        show_pcd = None
        AM_pcd = None
        am_color_value = []
        for f_id, one_feature in self.feature_server.items():
            pcd = None
            pcd_am = None

            for k_id, obs in one_feature.observation.items():
                ob = copy.copy(obs)
                if f_id in key_frame.include_feature_id_list:
                    # feature_color = plt.get_cmap('tab20b')(f_id-min(key_frame.include_feature_id_list)+1)[:3]
                    # colors = np.tile(feature_color, (len(ob.colors), 1))
                    ob_map_value = None
                    t_base = cams[0].timestamp
                    pts = np.array(copy.copy(ob.points))

                    for cam in cams:
                        distances = np.linalg.norm(np.cross(pts - cam.gaze_vector_in_world[0], 
                                                   pts - cam.gaze_vector_in_world[1]), axis=-1)
                        distances /= np.linalg.norm(cam.gaze_vector_in_world[0]-cam.gaze_vector_in_world[1])
                        
                        dis2 = distances**2
                        sigma = 0.3
                        lbda = 0
                        each_map_value = np.exp(-dis2/(2*sigma**2))*np.exp(-lbda*(cam.timestamp-t_base))

                        if ob_map_value is None:
                            ob_map_value = each_map_value
                        else:
                            ob_map_value += each_map_value

                    ob_map_value = np.array(ob_map_value)
                    # feature_map_value /=np.sum(feature_map_value)
                    # print(np.max(feature_map_value),np.min(feature_map_value))
                    # norm_map_value = (ob_map_value - np.min(ob_map_value)) / (np.max(ob_map_value) - np.min(ob_map_value))
                    # colors= plt.cm.jet(norm_map_value)[:, :-1]
                    # ob.colors = o3d.utility.Vector3dVector(colors)
                    if pcd_am is None:
                        pcd_am = ob
                    else:
                        pcd_am += ob
                    am_color_value.append(ob_map_value)
                else:
                    cmap = plt.get_cmap('tab20')(15)[:3]
                    colors = np.tile(cmap, (len(ob.colors), 1))
                    ob.colors = o3d.utility.Vector3dVector(colors)
                    if pcd is None:
                        pcd = ob
                    else:
                        pcd += ob

            # fuse_terrians = np.array(pcd.points)
            # dist_A = fuse_terrians[:,2] - np.mean(fuse_terrians[:,2])
            # pcd = pcd.select_by_index(np.arange(len(fuse_terrians))[dist_A < np.mean(dist_A) + 1 * np.std(dist_A)])
            if show_pcd is None:
                show_pcd = pcd
            elif pcd is not None:
                show_pcd += pcd
            if AM_pcd is None:
                AM_pcd = pcd_am
            elif pcd_am is not None:
                AM_pcd += pcd_am

        # am_color_value = np.array(am_color_value).reshape(-1)\
        am_color_value = np.concatenate([np.array(item) for item in am_color_value])
        norm_map_value = (am_color_value - np.min(am_color_value)) / (np.max(am_color_value) - np.min(am_color_value))
        colors= plt.cm.jet(norm_map_value)[:, :-1]
        AM_pcd.colors = o3d.utility.Vector3dVector(colors)
        show_pcd += AM_pcd

        points = np.asarray(show_pcd.points)
        points[:, -1] += 0.01
        show_pcd.points = o3d.utility.Vector3dVector(points)
        line_set = o3d.geometry.LineSet() 
        line_points = [] 
        lines = []
        spheres = []
        for cam in cams:
            # print(cam.id)
            start_point = cam.gaze_vector_in_world[1].reshape(-1)
            end_point = cam.gaze_in_world

            z_to_gaze_mat = np.identity(4)  # 创建单位矩阵，用于变换
            z_vec = end_point - start_point  # 计算两点之间的方向向量
            z_to_gaze_mat[:3, 2] = z_vec / np.linalg.norm(z_vec)  # 归一化并设置为z方向
            z_to_gaze_mat[:3, 0] = np.cross(z_to_gaze_mat[:3, 1], z_to_gaze_mat[:3, 2])  # 计算x方向的向量
            z_to_gaze_mat[:3, 0] /= np.linalg.norm(z_to_gaze_mat[:3, 0])  # 归一化x方向向量
            z_to_gaze_mat[:3, 3] = (start_point + end_point) / 2  # 设置圆柱体的中心位置

            height = np.linalg.norm(start_point - end_point)  # 计算圆柱体的高度
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=height)  # 创建圆柱体网格
            mesh_cylinder.paint_uniform_color([1,0,0])  # 为圆柱体上色
            mesh_cylinder_origin = copy.deepcopy(mesh_cylinder)  # 复制原始圆柱体网格
            mesh_cylinder.transform(z_to_gaze_mat)  # 对圆柱体应用变换矩阵

            idx = len(line_points)
            line_points.append(start_point.tolist())
            line_points.append(end_point.tolist())
            lines.append([idx, idx + 1])
            # spheres.append(mesh_cylinder)q

        # self.open3d.view_pcd_with_axis(segmented_pcd+cloud)

        pcd_tree = o3d.geometry.KDTreeFlann(segmented_pcd+cloud)
        radius = 0.25
        [k, idx, _] = pcd_tree.search_radius_vector_3d(key_frame.cam_state.gaze_in_world, radius)
        cloud_colored,map_color,weighted_gaze = self.attention_map_plotter(segmented_pcd+cloud,idx,
                                                         key_frame.cam_state.gaze_vector_in_world,sigma = 0.5)
        self.open3d.view_pcd_with_axis(cloud_colored)#1
        self.open3d.view_pcd_with_axis(segmented_pcd)#2
        self.open3d.view_pcd_with_axis(show_pcd,line_set,spheres)      

    def attention_map_plotter(self,pcd,idx,gaze_vector_in_world,sigma = 0.01):
        pcd_colored = copy.copy(pcd)
        
        map_color_pcd = copy.copy(pcd)
        pcd_points = np.array(pcd.points)
        pcd_points = pcd_points[idx]
        map_value=np.zeros(len(pcd_points))

        distances = np.linalg.norm(np.cross(pcd_points - gaze_vector_in_world[0], 
                                                   pcd_points - gaze_vector_in_world[1]), axis=-1)
        distances /= np.linalg.norm(gaze_vector_in_world[0]-gaze_vector_in_world[1])
        gaze_point = pcd_points[np.argmin(distances)]
        dis2 = np.linalg.norm(pcd_points - gaze_point, axis=1)**2
        map_value = np.exp(-dis2/(2*sigma**2))
        map_value = map_value/np.sum(map_value)
        weight_points = pcd_points*map_value[:,np.newaxis]
        gaze = np.sum(weight_points,axis=0)

        norm_map_value = (map_value - np.min(map_value)) / (np.max(map_value) - np.min(map_value))
        colors= plt.cm.jet(norm_map_value)[:, :-1]

        colors2 = np.array(pcd_colored.colors)
        colors2[idx] = colors
        pcd_colored.colors = o3d.utility.Vector3dVector(colors2)

        return pcd_colored,colors,gaze

    def report_generator(self):
        exp_idx = 1

        report = {
            "Setup": {
                "subject": "czk",
                "exp_mode": 1,
                "total_steps": len(self.gait_server),
            },
            "Gait_info": {f"step {i + 1}": state.to_dict() 
                            for i, state in enumerate(self.gait_server)}
        }
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, f"exp_report/walking_report{exp_idx}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(report, f, indent=2, sort_keys=False)

    def fuse_video(self,step_cams,k_ids,is_dir = 0):

        self.key_frame_server.clear()  # 清空现有字典
        self.feature_server.clear()
        self.init_ground = None
        self.next_feature_id = 0
        self.is_feature_init = False
        self.cam_server = []
        self.key_frame_id_list = []
        self.min_dis = 0.05
        self.min_num = 100
        self.ground_threshold = 0.018

        frame_paths = []
        cameras = []
        camera_positions = []
        traj = o3d.geometry.LineSet()
        lines = []
        l_id = 0
        gray = plt.get_cmap('tab20')(15)[:3]

        output_dir = "pcd_frames"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, output_dir)
        os.makedirs(full_path, exist_ok=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080,visible=True)
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        default_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        intrinsic = default_params.intrinsic

        pcd = None
        show_pcd = None
        AM_pcd = None
        ground_pcd = None
        fp_points = []
        gaze_lists = []
        fp_traj = []

        for step,cams in enumerate(step_cams):
            slide_cams = []
            # vis.clear_geometries()
                
            for cam_state in cams:
                vis.clear_geometries()
                slide_cams.append(cam_state)

                if cam_state.id == k_ids[step]:
                    # vis.clear_geometries()
                    _,cloud,_,_ = self.gaze.tracker_to_rgbd_without_depth(cam_state.gaze_point,cam_state,5,cam_state.mask)
                    cloud = cloud.transform(cam_state.transform_matrix)
                    cam_state.pcd_in_world = cloud
                    key_frame, segmented_pcd, segmented_labels, center_position,foothold_types = self.add_new_key_frame(cam_state)
                    segmented_terrain_pcd, segmented_labels, center_position, foothold_types = self.tracking_features(segmented_pcd, 
                                                                                                segmented_labels, center_position,foothold_types,key_frame,to_lib=False)
                    self.add_new_features(segmented_terrain_pcd, segmented_labels, center_position,foothold_types,key_frame)

                    am_color_value = []
                    weighted_pcd = []
                    show_pcd = None
                    AM_pcd = None
                    for f_id, one_feature in self.feature_server.items():
                        pcd = None
                        pcd_am = None
                        each_fea_value = []
                        fea_pts = []

                        for k_id, obs in one_feature.observation.items():
                            ob = copy.copy(obs)
                            if f_id in key_frame.include_feature_id_list:
                                ob_map_value = None
                                t_base = cams[0].timestamp
                                pts = np.array(copy.copy(ob.points))
                                fea_pts.append(pts)

                                for cam in slide_cams:
                                    distances = np.linalg.norm(np.cross(pts - cam.gaze_vector_in_world[0], 
                                                            pts - cam.gaze_vector_in_world[1]), axis=-1)
                                    distances /= np.linalg.norm(cam.gaze_vector_in_world[0]-cam.gaze_vector_in_world[1])
                                    
                                    dis2 = distances**2
                                    sigma = 0.5
                                    lbda = 0
                                    each_map_value = np.exp(-dis2/(2*sigma**2))
                                    # *np.exp(-lbda*(cam.timestamp-t_base))
                                    if ob_map_value is None:
                                        ob_map_value = each_map_value
                                    else:
                                        ob_map_value += each_map_value
                                ob_map_value = np.array(ob_map_value)
                                if pcd_am is None:
                                    pcd_am = ob
                                else:
                                    pcd_am += ob
                                am_color_value.append(ob_map_value)
                                each_fea_value.append(ob_map_value)
                                
                            else:
                                if pcd is None:
                                    pcd = ob
                                else:
                                    pcd += ob
                    
                        if show_pcd is None:
                            show_pcd = pcd
                        elif pcd is not None:
                            show_pcd += pcd
                        if AM_pcd is None:
                            AM_pcd = pcd_am
                        elif pcd_am is not None:
                            AM_pcd += pcd_am

                        if len(each_fea_value)!=0:
                            each_fea_value = np.concatenate([np.array(item) for item in each_fea_value])
                            fea_pts = np.concatenate([np.array(item) for item in fea_pts])
                            norm_value = each_fea_value/np.sum(each_fea_value)
                            weighted_pcd.append(fea_pts*norm_value.reshape(-1,1))

                    mean_map_value = np.array([sum(row)/len(row) for row in am_color_value])
                    fea_pcd = weighted_pcd[np.argmax(mean_map_value)]
                    fpp = np.sum(fea_pcd,axis=0)
                    # print('fpp:',fpp)
                    fp_points.append(fpp)
                    gaze_lists.append(cam_state.gaze_in_world)
                    
                    if step >=1:

                        start_point = copy.copy(fp_points[step-1])
                        end_point = copy.copy(fp_points[step])
                        start_point[-1]+=0.03
                        end_point[-1]+=0.03

                        z_to_gaze_mat = np.identity(4)
                        z_vec = end_point - start_point
                        z_to_gaze_mat[:3, 2] = z_vec / np.linalg.norm(z_vec)
                        z_to_gaze_mat[:3, 0] = np.cross(z_to_gaze_mat[:3, 1], z_to_gaze_mat[:3, 2])
                        z_to_gaze_mat[:3, 0] /= np.linalg.norm(z_to_gaze_mat[:3, 0])
                        z_to_gaze_mat[:3, 3] = (start_point + end_point) / 2 

                        height = np.linalg.norm(start_point - end_point)  # 计算圆柱体的高度
                        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=height)
                        mesh_cylinder.paint_uniform_color([1,0,0])
                        mesh_cylinder.transform(z_to_gaze_mat)  # 对圆柱体应用变换矩阵
                        fp_traj.append(mesh_cylinder)

                    am_color_value = np.concatenate([np.array(item) for item in am_color_value])
                    norm_map_value = (am_color_value - np.min(am_color_value)) / (np.max(am_color_value) - np.min(am_color_value))
                    colors= plt.cm.jet(norm_map_value)[:, :-1]
                    AM_pcd.colors = o3d.utility.Vector3dVector(colors)

                    if ground_pcd is None:
                        ground_pcd = key_frame.cam_state.ground_cloud
                    else:
                        ground_pcd += key_frame.cam_state.ground_cloud

                if AM_pcd is not None:
                    if show_pcd is not None:
                        vis.add_geometry(AM_pcd+show_pcd)
                    else:
                        vis.add_geometry(AM_pcd)
                    vis.add_geometry(ground_pcd)

                cam_pose = cam_state.transform_matrix.copy()
                extrinsic = np.linalg.inv(cam_pose)
                camera_position = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                camera_positions.append(camera_position)

                if len(camera_positions)>=2:
                    lines.append([l_id,l_id+1])
                    traj.points = o3d.utility.Vector3dVector(camera_positions)
                    traj.lines = o3d.utility.Vector2iVector(lines)
                    traj.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.0, 0.0]), (len(lines), 1)))
                    l_id += 1

                if cam_state.id <= k_ids[step]:
                    camera_ls = o3d.geometry.LineSet.create_camera_visualization(intrinsic, extrinsic, scale = 0.1)
                    camera_ls.paint_uniform_color([0,1,0])
                    cameras.append(camera_ls)

                vis.add_geometry(world_axes)
                for camera in cameras:
                    vis.add_geometry(camera)
                vis.add_geometry(traj)
                if len(fp_points) !=0:
                    if len(fp_traj)!=0:
                        for tj in fp_traj:
                            vis.add_geometry(tj)
                    for point in fp_points:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                        sphere.translate(point)
                        sphere.paint_uniform_color([1.0, 1.0, 0.0])
                        vis.add_geometry(sphere)
                    for gaze in gaze_lists:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                        sphere.translate(gaze)
                        sphere.paint_uniform_color([0.0, 1.0, 0.0])
                        vis.add_geometry(sphere)

                param = o3d.camera.PinholeCameraParameters() 
                param.intrinsic = intrinsic 
                extrinsic_original = np.array([ [1, 0, 0, 0], [0, -1, 0, 0.55], [0, 0, -1, 2], [0, 0, 0, 1] ])
                theta = radians(-45) # 角度转弧度 
                R_x = np.array([ [1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)] ]) 
                R_original = extrinsic_original[:3, :3] 
                t_original = extrinsic_original[:3, 3] 
                R_new = R_original @ R_x # 构建新外参 
                extrinsic_new = np.eye(4) 
                extrinsic_new[:3, :3] = R_new 
                extrinsic_new[:3, 3] = t_original 
                param.extrinsic = extrinsic_new 

                ctr = vis.get_view_control()
                param2 = o3d.io.read_pinhole_camera_parameters(self.base_dir + "/viewer.json")
                param2.intrinsic = intrinsic
                ctr.convert_from_pinhole_camera_parameters(param2, allow_arbitrary=True)

                vis.poll_events()
                vis.update_renderer()

                img_path = os.path.join(full_path, f"k_id_{cam_state.id}.png")
                vis.capture_screen_image(img_path)
                frame_paths.append(img_path)

        img_array = []
        fps = 11
        for img_path in frame_paths:
            img = cv2.imread(img_path)  # 读取每张图像
            height, width, layers = img.shape  # 获取图像的尺寸
            size = (width, height)  # 设置视频的尺寸
            img_array.append(img)  # 将图像添加到数组中

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用H.264编码器创建视频写入器
        out = cv2.VideoWriter(full_path+'/1.avi', fourcc, fps, size)  # 初始化视频写入器

        for i in range(len(img_array)):
            out.write(img_array[i])  # 写入每一帧图像到视频

        vis.run()
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("/home/czk/catkin_ws/src/samcon_perception/scripts/GPP/viewer.json", param)

    def release_resources(self):
        # print(self.acc)
        self.cam_server_temp = self.cam_server
        cv2.destroyAllWindows()

if __name__ == "__main__":

    rospy.init_node("fpp_node")
    
    fusion = Fusion(maker = 0) 

    rospy.on_shutdown(fusion.release_resources)
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        fusion.display_image()
        rate.sleep()

    print('Processing different cam states...')

    # fusion.post_processing()

    fusion.show_feature_by_feature()

    # if not fusion.maker:
    #     fusion.show_fused_pcd()
