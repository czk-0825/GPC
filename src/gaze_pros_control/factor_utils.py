import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import pyransac3d as pyrsc
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
import sys
import os
import rospkg
# sys.stdout = open(os.devnull, 'w')
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

default_cam_intri = np.asarray(
    [[513.88818359375,   0.        , 318.6736755371094],
    [0.        , 513.88818359375 , 239.4502410888672],
    [0.        ,   0.        ,   1.        ]])

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('gaze_pros_control')
base_dir = os.path.join(pkg_path, 'scripts/env_perception')
rgbd_tracker_paras = np.load(base_dir + '/rgbd_tracker_paras.npy', allow_pickle=True).item()

class Open3d_utils():
    def __init__(self):

        self.model = 'pinhole'
        self.width = 640
        self.height = 480
        self.fps = 30
        self.rgbd_tracker_paras = rgbd_tracker_paras 

    def read_cam_intrinsic(self):
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.width = self.width
        pinhole_camera_intrinsic.height = self.height
        pinhole_camera_intrinsic.intrinsic_matrix = default_cam_intri
        return pinhole_camera_intrinsic

    def read_rgbd_pcd(self,rgb_img, depth_img):
        pinhole_camera_intrinsic = self.read_cam_intrinsic()
        current_color = o3d.geometry.Image(rgb_img)
        current_depth = o3d.geometry.Image(depth_img)
        current_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            current_color, current_depth, depth_trunc=7.5, convert_rgb_to_intensity=False)
        current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            current_rgbd_image, pinhole_camera_intrinsic)
        return current_pcd
    
    def calc_cloud_in_ground(self,cloud_pcd,ground =0.016):
        tt = time.time()
        plane_model, ground_indices = cloud_pcd.segment_plane(
            distance_threshold=ground, ransac_n=3, num_iterations=100)
        # print(f'RANSAC_time:{time.time()-tt}')
        cloud_ground = np.array(cloud_pcd.points)[ground_indices]
        z_ground = -np.mean(cloud_ground[..., 2])
        current_in_ground,d = self.calc_transformation_from_plane_model(plane_model, offset_y=True)
        return current_in_ground, plane_model, ground_indices,z_ground
    
    def calc_transformation_from_plane_model(self,plane_model, offset_y=True):
        [a, b, c, d] = plane_model
        # print(d)
        z_now = np.asarray([a, b, c])
        z_now = z_now / np.linalg.norm(z_now)
        y_now = np.asarray([0, 1, 0])
        x_now = np.cross(y_now, z_now)
        if offset_y:
            y_now = np.cross(z_now, x_now)
        coordinate_system_now = np.c_[x_now.reshape((3, 1)),
                                    y_now.reshape((3, 1)),
                                    z_now.reshape((3, 1))]
        rot_mat = np.linalg.inv(coordinate_system_now)
        transform_mat = np.identity(4)
        transform_mat[:3, :3] = rot_mat
        return transform_mat,d
    
    def remove_cloud_background(self, cloud_pcd,plane_model, ground_indices, remove_ground=True):
        outlier_indices ,z_ground= self.calc_outliers_of_terrain(cloud_pcd, ground_indices, plane_model)
        if remove_ground:
            terrain_indices = set(np.arange(len(cloud_pcd.points)).tolist())- set(ground_indices)\
                            - set(outlier_indices.tolist())
        else:
            terrain_indices = set(np.arange(len(cloud_pcd.points)).tolist())\
                            - set(outlier_indices.tolist())
        terrain_indices = np.asarray(list(terrain_indices))
        if 0 == len(terrain_indices):
            terrain_indices = 0
        ground_cloud = copy.deepcopy(cloud_pcd)
        ground_cloud = ground_cloud.select_by_index(ground_indices)
        cloud_pcd = copy.deepcopy(cloud_pcd)
        cloud_no_ground = np.asarray(cloud_pcd.points)[terrain_indices]
        valid_indices = cloud_no_ground[..., 2] < 0
        cloud_pcd.points = o3d.utility.Vector3dVector(cloud_no_ground[valid_indices])
        cloud_pcd.colors = o3d.utility.Vector3dVector(np.asarray(cloud_pcd.colors)[terrain_indices][valid_indices])
        return cloud_pcd, plane_model, ground_cloud, z_ground
    
    def calc_outliers_of_terrain(self,cloud_pcd, ground_indices, plane_model):
        cloud_pcd_temp = copy.deepcopy(cloud_pcd)
        cloud = np.asarray(cloud_pcd_temp.points)
        cloud_ground = cloud[ground_indices]
        z_ground = np.mean(cloud_ground[..., 2])
        # print(z_ground)
        outliers = np.bitwise_or(cloud[..., 2] < z_ground - 0.25, cloud[..., 2] > z_ground + 0.2)
        outlier_indices = np.nonzero(outliers)[0]
        return outlier_indices ,z_ground
    
    def label_connected_area(self,keyframe,dis = 0.08,min = 30):
        print('------------------------------------------------')
        cloud_pcd = keyframe.pcd_calib_no_ground.transform(np.linalg.inv(keyframe.calibr_matrix))
        t = time.time()
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            eps = dis  # 聚类的epsilon参数，决定相邻点的距离
            min_points = min  # 聚类的最小点数
            labels = np.array(cloud_pcd.cluster_dbscan(eps=eps, min_points=min_points))  # 使用DBSCAN算法进行聚类
        '''
        DBSCAN [Ester1996] that is a density based clustering algorithm
        '''
        cloud_pcd.points = o3d.utility.Vector3dVector(np.asarray(cloud_pcd.points)[labels >= 0])  # 过滤无效的点
        cloud_pcd.colors = o3d.utility.Vector3dVector(np.asarray(cloud_pcd.colors)[labels >= 0])
        labels = labels[labels >= 0]
        if len(labels) == 0:
            print('=============================================')
            return [],[],[],[]
        # labels = self.sort_clusters(cloud_pcd, labels)  # 对聚类结果进行排序
        print(f"point cloud has {labels.max() + 1} clusters with {time.time()-t}")  # 打印聚类结果中的簇数
        # colors = plt.get_cmap("tab20")(labels / 20)  # 使用颜色映射标记不同的簇
        # cloud_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # 应用颜色
        

        t = time.time()
        cluster_plane, plane_labels,plane_angles = self.clusters_plane_type(cloud_pcd,labels)
        if len(plane_labels) == 0:
            print('=============================================')
            return [],[],[],[]
        center_mat = np.zeros((len(np.unique(plane_labels)), 3))  # 初始化每个聚类的中心点矩阵
        for i,label in enumerate(np.unique(plane_labels)):
            center_mat[i] = np.mean(np.array(cluster_plane.points)[plane_labels == label], axis=0)
        # print(f'type_time{time.time()-t}')
        return cluster_plane, plane_labels,plane_angles,center_mat  # 返回带有颜色的点云和标签
    
    def sort_clusters(self,cloud_pcd, labels):
        points = np.asarray(cloud_pcd.points)  # 获取点云中的点
        center_mat = np.zeros((np.max(labels) + 1, 3))  # 初始化每个聚类的中心点矩阵
        for i in range(np.max(labels) + 1):
            center_mat[i] = np.mean(points[labels == i], axis=0)  # 计算每个聚类的中心点
        indices = np.argsort(center_mat[:, 0])  # 根据中心点的X坐标进行排序
        sorted_labels = np.zeros(labels.shape, int)  # 初始化排序后的标签
        for i in range(len(indices)):
            sorted_labels[labels == indices[i]] = i  # 根据排序结果重新标记聚类
        return sorted_labels  # 返回排序后的标签

    def clusters_plane_type(self,cloud_pcd,labels,):
        plane_pcd = None
        plane_labels = []
        plane_angles = []

        for cluster_id in np.unique(labels):
            idx_cluster = np.where(labels == cluster_id)[0]
            cluster_points = cloud_pcd.select_by_index(idx_cluster)
            point_labels = labels[idx_cluster]
            t = time.time()
            plane_model, inliers = cluster_points.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=20)
            # print(f'RANSAC time:{time.time()- t}')
            normal = plane_model[:-1]
            vertical = np.array([0, 0, 1])
            angle = np.arccos(np.dot(normal, vertical) / (np.linalg.norm(normal) * np.linalg.norm(vertical)))
            angle = -np.sign(normal[1])*np.degrees(angle)
            inlier_cloud = cluster_points.select_by_index(inliers)   
            plane_point_labels = point_labels[inliers]
            inlier_ratio = len(inliers) / len(cluster_points.points)
            if inlier_ratio > 0.1 and abs(angle) < 30:
                plane_labels.append(plane_point_labels)
                plane_angles.append(angle)
                if plane_pcd is None:
                    plane_pcd = inlier_cloud
                else:
                    plane_pcd += inlier_cloud
        if len(plane_labels) != 0:
            return plane_pcd, np.concatenate(plane_labels, axis=0),plane_angles
        else:
            return [],[],[]
    
    def view_pcd_with_axis(self,pcd,line = None,spheres = None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # world_axes.translate(np.mean(np.asarray(pcd.points) + np.asarray([0, 1, -0.05]), axis=0))
        vis.add_geometry(pcd)
        vis.add_geometry(world_axes)
        if line is not None:
            vis.add_geometry(line)
        if spheres is not None:
            for sphere in spheres:
                vis.add_geometry(sphere)

        # ctr = vis.get_view_control()
        # param2 = o3d.io.read_pinhole_camera_parameters("/home/czk/catkin_ws/src/samcon_perception/scripts/GPP/camera.json")
        # ctr.convert_from_pinhole_camera_parameters(param2, allow_arbitrary=True)
        vis.run()
        # ctr = vis.get_view_control()
        # param = ctr.convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters("/home/czk/catkin_ws/src/samcon_perception/scripts/GPP/viewer.json", param)

class Gaze_utils():
    def __init__(self):

        self.op = Open3d_utils()
        self.default_cam_intri = default_cam_intri
        self.rgbd_tracker_paras = rgbd_tracker_paras
        self.flag = 0
        
    def uv_vec2cloud(self,uv_vec, depth_img, depth_scale = 1e-3):
        '''
            uv_vec: n×2,
            depth_image: rows * cols
        '''
        fx = default_cam_intri[0, 0]
        fy = default_cam_intri[1, 1]
        cx = default_cam_intri[0, 2]
        cy = default_cam_intri[1, 2]
        cloud = np.zeros((len(uv_vec), 3))
        cloud[:, 2] = depth_img[uv_vec[:, 1].astype(int), uv_vec[:, 0].astype(int)] * depth_scale
        cloud[:, 0] = (uv_vec[:, 0] - cx) * (cloud[:, 2] / fx)
        cloud[:, 1] = (uv_vec[:, 1] - cy) * (cloud[:, 2] / fy)
        return cloud

    def depth2cloud(self,img_depth, cam_intri_inv = None):
        # default unit of depth and point cloud is mm.
        if cam_intri_inv is None:
            cam_intri_inv = np.zeros((1, 1, 3, 3))
            cam_intri_inv[0, 0] = np.linalg.inv(default_cam_intri)
        uv_vec = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0))[:, :, [1, 0]].reshape((-1, 2))
        point_cloud = self.uv_vec2cloud(uv_vec, img_depth, depth_scale=1)
        return point_cloud

    def cam1_to_cam2_xyz(self,uv_vec_1, z_vec_1, cam1_matrix, T_cam1_in_cam2):
        uv_vec_1 = uv_vec_1.reshape((-1, 2))
        num_points = uv_vec_1.shape[0]
        uv_vec_1 = np.asarray(uv_vec_1)
        uvz_vec_1 = np.ones((num_points, 3))
        uvz_vec_1[:, :2] = uv_vec_1
        uvz_vec_1 *= np.reshape(z_vec_1, (-1, 1))
        p_vec_1 = np.ones((4, num_points))
        p_vec_1[:3, :] = np.matmul(np.linalg.inv(cam1_matrix), uvz_vec_1.T)
        p_vec_2 = np.matmul(T_cam1_in_cam2, p_vec_1)
        # p_vec_2[1] += 200
        return p_vec_2

    def tracker_to_rgbd_without_depth(self,uv_tracker, cam,down_sample_rate,mask):
        self.flag+=1
        start = time.time()
        img_rgb = cam.img_rgb
        img_depth = cam.img_depth

        dep = img_depth.flatten()
        valid_indices = np.where((dep != 0) & (dep <= 7500))[0]

        uv_tracker = uv_tracker.reshape([1, -1])
        cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
        T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
        # T_tracker_in_depth[1,-1] += 0
        # T_tracker_in_depth[0,-1] -= 75
        # T_tracker_in_depth[1,-1] += 150
        if self.flag <15:
            T_tracker_in_depth[1,-1] += 200
        elif self.flag>=15 and self.flag<40:
            T_tracker_in_depth[1,-1] += 150
        elif self.flag>=40 and self.flag<100:
            T_tracker_in_depth[1,-1] += 50
        else:
            T_tracker_in_depth[1,-1] += 100

        open_cloud = self.op.read_rgbd_pcd(img_rgb, img_depth)
        original_indices = np.arange(len(dep))[valid_indices][::down_sample_rate]
        open_cloud = open_cloud.uniform_down_sample(down_sample_rate)

        points_in_depth = np.array(copy.copy(open_cloud.points))*1000
        p_eye_in_depth = self.cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                        cam1_matrix=cam_matrix_tracker,
                                        T_cam1_in_cam2=T_tracker_in_depth)[:3].T
        p_virtual_point_in_depth = self.cam1_to_cam2_xyz(uv_tracker, np.asarray([2500]),
                                                    cam1_matrix=cam_matrix_tracker,
                                                    T_cam1_in_cam2=T_tracker_in_depth)[:3].T

        distance = np.linalg.norm(np.cross(points_in_depth - p_eye_in_depth,
                                        points_in_depth-p_virtual_point_in_depth), axis=-1)
        distance /= np.linalg.norm(p_eye_in_depth - p_virtual_point_in_depth, axis=-1)
        index_min = original_indices[np.argmin(distance)]
 
        v = int(np.floor(index_min / img_depth.shape[1]))
        u = index_min - v * img_depth.shape[1]
        uv_est_in_depth = np.asarray([u, v])
        gaze_in_camera = self.get_point_in_camera(uv_est_in_depth,img_depth)
        # print('Computing time of 2D gaze to 3D gaze: {:0f} ms'.format(1000 * (time.time() - start)))
        virtual_sight = np.array([p_virtual_point_in_depth , p_eye_in_depth]) * 1e-3
        
        if cam.mask is not None:
            masked_points = mask.flatten()[original_indices] == 255
            valid_indices = np.arange(len(original_indices))[np.where(~masked_points)[0]]
            open_cloud = open_cloud.select_by_index(valid_indices)

        return uv_est_in_depth, open_cloud, virtual_sight,gaze_in_camera

    def get_point_in_camera(self,gaze_in_depth,img_depth):
        gaze_in_camera = []
        u,v = gaze_in_depth
        depth_value = img_depth[int(v), int(u)]
        fx = self.default_cam_intri[0, 0]
        fy = self.default_cam_intri[1, 1]
        cx = self.default_cam_intri[0, 2]
        cy = self.default_cam_intri[1, 2]
        z_cam = depth_value * 1e-3  
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        gaze_in_camera = np.array([x_cam, y_cam, z_cam])
        return gaze_in_camera

    def get_valid_depth(self,depth_map, u, v, max_radius=5):
        """螺旋搜索邻域，直到找到非零深度值，并返回新的坐标"""
        h, w = depth_map.shape
        u, v = int(u), int(v)
        
        if u < 0 or u >= w or v < 0 or v >= h:
            return 0, u, v
        
        depth_value = depth_map[v, u]
        if depth_value != 0:
            return depth_value, u, v
        
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    # 只搜索外圈（避免重复搜索内层）
                    if abs(dx) == r or abs(dy) == r:
                        new_u, new_v = u + dx, v + dy
                        if 0 <= new_u < w and 0 <= new_v < h:
                            neighbor_depth = depth_map[new_v, new_u]
                            if neighbor_depth != 0:
                                return neighbor_depth, new_u, new_v
        return 0, u, v  # 未找到有效值，返回原坐标

    def get_point_in_world(self,u,v,cam):
        R_world_camera = cam.transform_matrix[:3,:3]
        T_world_camera = cam.transform_matrix[:3,-1]
        depth_value = cam.img_depth[int(v), int(u)]
        if depth_value == 0:
            depth_value,u,v = self.get_valid_depth(cam.img_depth, u, v)
            if not depth_value:
                print('00000000000000000000')
        
        fx = self.default_cam_intri[0, 0]
        fy = self.default_cam_intri[1, 1]
        cx = self.default_cam_intri[0, 2]
        cy = self.default_cam_intri[1, 2]
        z_cam = depth_value * 1e-3  
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        pc = np.array([x_cam, y_cam, z_cam])
        pw = np.matmul(R_world_camera, pc) + T_world_camera
        return pw

    def uvz2xyz(self,u, v, d, cam_intri_inv= None):
        if cam_intri_inv is None:
            cam_intri_inv = np.linalg.inv(default_cam_intri)
        uvd_vec = np.asarray([u, v , 1]) * d
        xyz = np.matmul(cam_intri_inv, uvd_vec)
        return xyz

    def time_attention_map(self,pcd,distances,time_factors,time_window_size):
        sigma=0.1 
        map_value=np.zeros(len(distances))
        dis2 = distances**2
        map_value = np.exp(-dis2/(2*sigma**2)) * (time_factors+2)/(time_window_size+2)
        # print('time:',(time_factors+2)/(time_window_size+2))
        map_value = map_value/np.sum(map_value)
        weight_points = pcd*map_value[:,np.newaxis]
        gaze_point = np.sum(weight_points,axis=0)
        # colors = plt.cm.jet(all_img_value/ np.max(all_img_value))
        return gaze_point, map_value
