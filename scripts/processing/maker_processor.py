import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter1d
from detecta import detect_peaks
from scipy.spatial.distance import cdist
import yaml
from matplotlib.patches import Patch
from scipy.linalg import svd
import os
import scienceplots
from matplotlib.lines import Line2D
from detecta import detect_peaks
import scipy.stats as stats

class Maker_processor():
    def __init__(self,sub,base_dir):

        self.sub = sub
        self.base_dir = base_dir
        self.model = 'pinhole'
        self.width = 640
        self.height = 480
        self.fps = 30
        self.depth_image = None
        self.rgb_image = None
        self.pose = None
        self.calibr_pw = []
        
        self.default_cam_intri = np.asarray(
            [[513.88818359375,   0.        , 318.6736755371094],
             [0.        , 513.88818359375 , 239.4502410888672],
             [0.        ,   0.        ,   1.        ]]
        )
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.pinhole_camera_intrinsic.width = self.width
        self.pinhole_camera_intrinsic.height = self.height
        self.pinhole_camera_intrinsic.intrinsic_matrix = self.default_cam_intri

    def create_point_cloud(self, depth_image, rgb_image):

        self.depth_image = depth_image
        self.rgb_image = rgb_image
        
        depth = o3d.geometry.Image(depth_image.astype(np.float32))
        color = o3d.geometry.Image(rgb_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=8, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.pinhole_camera_intrinsic)
        dep = depth_image.flatten()
        valid_indices = np.where((dep != 0) & (dep <= 8000))[0]

        o3d.visualization.draw_geometries([pcd])

        return pcd, valid_indices

    def on_mouse_click(self, event, x, y, flags, param):
        """鼠标点击事件处理函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_image is not None:
                depth_value = self.depth_image[y, x]  # 注意OpenCV的坐标顺序是(y,x)
                pw = self.get_point_in_world(x,y)
                print(f"Clicked at pixel (x={x}, y={y}) - point: {pw}")
                
                # 在图像上显示点击位置和深度值
                display_img = self.rgb_image.copy()
                cv2.circle(display_img, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(display_img, f"Depth: {depth_value}mm", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("RGB Image with Depth", display_img)
                self.calibr_pw.append(pw)
                if len(self.calibr_pw) > 5:#6:
                    self.calibr_pw = self.calibr_pw[1:]

    def get_point_in_world(self,u,v):
        R_world_camera = self.pose[:3,:3]
        T_world_camera =self.pose[:3,-1]
        depth_value = self.depth_image[int(v), int(u)]
        if depth_value == 0:
            pw = np.array([0,0,0]) 
        else:
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

    def process_and_display(self, depth_image, rgb_image,pose):
        """处理并显示点云,同时设置RGB图像的点击事件"""
        self.pose = pose
        pcd, idxs = self.create_point_cloud(depth_image, rgb_image)
        
        cv2.namedWindow("RGB Image with Depth")
        cv2.setMouseCallback("RGB Image with Depth", self.on_mouse_click)
        cv2.imshow("RGB Image with Depth", rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(self.calibr_pw)
        np.save(self.base_dir + f'/CZK/{self.sub}_pw_tracker.npy',self.calibr_pw)

def generate_vector_from_points(p1, p2, angle_deg, length):
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v_magnitude = np.linalg.norm(v)
        unit_v = v / v_magnitude
        angle_rad = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        rotated_vector = rotation_matrix @ unit_v
        result_vector = rotated_vector * length
        result_point = (p1[0] + result_vector[0], p1[1] + result_vector[1])
        return [*result_point,(p1[-1]+p2[-1])/2]

def draw_plane(plane_points, ax):
        # 用四个点连接出平面 (确保这是一个长方形)
        verts = plane_points  # plane_points 是四个点，连接为一个平面
        poly3d = Poly3DCollection(verts, facecolors='indianred', linewidths=1, edgecolors='black', alpha=0.6)
        ax.add_collection3d(poly3d)

def add_foot(ax, L_HEEL, L_TOE, R_HEEL, R_TOE, L_FOOT, R_FOOT, Maker_traj, eye_traj,gaze_data,frame_idx, l_line, r_line,l_leg,r_leg,hip,body,head,sight):
    L_KNEE = L_FOOT/2 + Maker_traj/2
    R_KNEE = R_FOOT/2 + Maker_traj/2
    L_ANKLE = L_FOOT + (L_KNEE - L_FOOT)/6
    R_ANKLE = R_FOOT + (R_KNEE - R_FOOT)/6
    HIP = L_KNEE/2+R_KNEE/2
    HEAD = eye_traj
    GAZE = gaze_data
    NECK = Maker_traj
    sight.set_data_3d(
            [HEAD[frame_idx, 0], GAZE[frame_idx, 0]],
            [HEAD[frame_idx, 1], GAZE[frame_idx, 1]],
            [HEAD[frame_idx, 2], GAZE[frame_idx, 2]]
        )
    hip.set_data_3d(
            [R_KNEE[frame_idx, 0], L_KNEE[frame_idx, 0]],
            [R_KNEE[frame_idx, 1], L_KNEE[frame_idx, 1]],
            [R_KNEE[frame_idx, 2], L_KNEE[frame_idx, 2]]
        )
    hip.set_data_3d(
            [R_KNEE[frame_idx, 0], L_KNEE[frame_idx, 0]],
            [R_KNEE[frame_idx, 1], L_KNEE[frame_idx, 1]],
            [R_KNEE[frame_idx, 2], L_KNEE[frame_idx, 2]]
        )
    body.set_data_3d(
            [HIP[frame_idx, 0], NECK[frame_idx, 0]],
            [HIP[frame_idx, 1], NECK[frame_idx, 1]],
            [HIP[frame_idx, 2], NECK[frame_idx, 2]]
        )
    head.set_data_3d(
            [HEAD[frame_idx, 0], NECK[frame_idx, 0]],
            [HEAD[frame_idx, 1], NECK[frame_idx, 1]],
            [HEAD[frame_idx, 2], NECK[frame_idx, 2]]
        )
    # l_line.set_data_3d(
    #         [L_HEEL[frame_idx, 0], L_TOE[frame_idx, 0]],
    #         [L_HEEL[frame_idx, 1], L_TOE[frame_idx, 1]],
    #         [L_HEEL[frame_idx, 2], L_TOE[frame_idx, 2]]
    #     )
    # r_line.set_data_3d(
    #         [R_HEEL[frame_idx, 0], R_TOE[frame_idx, 0]],
    #         [R_HEEL[frame_idx, 1], R_TOE[frame_idx, 1]],
    #         [R_HEEL[frame_idx, 2], R_TOE[frame_idx, 2]]
    #     )
    # l_leg.set_data_3d(
    #         [L_FOOT[frame_idx, 0], L_KNEE[frame_idx, 0]],
    #         [L_FOOT[frame_idx, 1], L_KNEE[frame_idx, 1]],
    #         [L_FOOT[frame_idx, 2], L_KNEE[frame_idx, 2]]
    #     )
    # r_leg.set_data_3d(
    #         [R_FOOT[frame_idx, 0], R_KNEE[frame_idx, 0]],
    #         [R_FOOT[frame_idx, 1], R_KNEE[frame_idx, 1]],
    #         [R_FOOT[frame_idx, 2], R_KNEE[frame_idx, 2]]
    #     )
    l_leg.set_data_3d(
            [L_ANKLE[frame_idx, 0], L_KNEE[frame_idx, 0]],
            [L_ANKLE[frame_idx, 1], L_KNEE[frame_idx, 1]],
            [L_ANKLE[frame_idx, 2], L_KNEE[frame_idx, 2]]
        )
    r_leg.set_data_3d(
            [R_ANKLE[frame_idx, 0], R_KNEE[frame_idx, 0]],
            [R_ANKLE[frame_idx, 1], R_KNEE[frame_idx, 1]],
            [R_ANKLE[frame_idx, 2], R_KNEE[frame_idx, 2]]
        )
    global poly3d1,poly3d2
    if poly3d1 is not None:
        poly3d1.remove()
        poly3d2.remove()
    verts = [[L_ANKLE[frame_idx],L_TOE[frame_idx],L_HEEL[frame_idx]]]  # plane_points 是四个点，连接为一个平面
    poly3d1 = Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='black', alpha=1)
    ax.add_collection3d(poly3d1)
    verts = [[R_ANKLE[frame_idx],R_TOE[frame_idx],R_HEEL[frame_idx]]]  # plane_points 是四个点，连接为一个平面
    poly3d2 = Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='black', alpha=1)
    ax.add_collection3d(poly3d2)

def estimate_pose_3d3d(points_src, points_dst):
    assert points_src.shape == points_dst.shape

    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst

    H = src_centered.T @ dst_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    
    return T

def calculate_reprojection_error(points_src, points_dst, T):
    ones = np.ones((points_src.shape[0], 1))
    points_src_homo = np.hstack((points_src, ones))
    points_transformed_homo = (T @ points_src_homo.T).T
    points_transformed = points_transformed_homo[:, :3]
    errors = np.linalg.norm(points_transformed - points_dst, axis=1)
    mean_error = np.mean(errors)
    return mean_error, errors, points_transformed

def find_window_above(data):
    x = np.arange(0,100,1)
    x_o  = np.arange(0,100,5)
    y_interp = np.interp(x, x_o, data)
    # print(y_interp,len(x),len(data))
    # plt.figure()
    # plt.plot(y_interp)
    # plt.show()
    start_idx = -1
    for i in range(len(data)):
        if data[i] > 90:
            start_idx = i
            break
    if start_idx == -1:
        return None, None
    end_idx = -1
    for i in range(len(data)-1, -1, -1):
        if data[i] > 90:
            end_idx = i
            break

    start = -1
    for i in range(len(y_interp)):
        if y_interp[i] > 90:
            start = i
            break
    if start == -1:
        return None, None
    end = -1
    for i in range(len(y_interp)-1, -1, -1):
        if y_interp[i] > 90:
            end = i
            break

    return start, end, start_idx, end_idx, y_interp

def main_maker():
    err_ab_gaze = []
    err_map_gaze = []
    for k in range(3):
        # sub = 'czk_2'
        sub = f'czk_{k+1}'
        base_dir = os.path.dirname(os.path.abspath(__file__))

        mp = Maker_processor(sub,base_dir)

        depth_image = cv2.imread(base_dir + f'/CZK/{sub}_depth.png', cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.imread(base_dir + f'/CZK/{sub}_rgb.png')
        pose = np.load(base_dir +  f'/CZK/{sub}_rot.npy')

        # mp.process_and_display(depth_image, rgb_image,pose)

        df = pd.read_csv(f'{base_dir}/CZK/250708/Trimmed_exp_{sub}.trc', delim_whitespace=True, skiprows=5, header=None)

        num_markers = (df.shape[1] - 2) // 3
        columns = ['Frame#', 'Time']
        maker_sets = ['l_heel','l_toe','r_heel','r_toe','t1.1','t1.2','t2.1','t2.2','t3.1','t3.2','t4.1','t4.2','init_l','init_r','ground','eye_l','eye_r']
        for i in range(0, num_markers):
            columns.extend([f'{maker_sets[i]}_X', f'{maker_sets[i]}_Y', f'{maker_sets[i]}_Z'])
        df.columns = columns

        eye_l_data = df[['eye_l_X', 'eye_l_Y', 'eye_l_Z']].values
        eye_r_data = df[['eye_r_X', 'eye_r_Y', 'eye_r_Z']].values
        t1_1 = df[['t1.1_X', 't1.1_Y', 't1.1_Z']].values
        t1_2 = df[['t1.2_X', 't1.2_Y', 't1.2_Z']].values
        t2_1 = df[['t2.1_X', 't2.1_Y', 't2.1_Z']].values
        t2_2 = df[['t2.2_X', 't2.2_Y', 't2.2_Z']].values
        t3_1 = df[['t3.1_X', 't3.1_Y', 't3.1_Z']].values
        t3_2 = df[['t3.2_X', 't3.2_Y', 't3.2_Z']].values
        t4_1 = df[['t4.1_X', 't4.1_Y', 't4.1_Z']].values
        t4_2 = df[['t4.2_X', 't4.2_Y', 't4.2_Z']].values
        ground_data = df[['ground_X', 'ground_Y', 'ground_Z']].values
        init_l_data = df[['init_l_X', 'init_l_Y', 'init_l_Z']].values
        init_r_data = df[['init_r_X', 'init_r_Y', 'init_r_Z']].values
        t1_1[:,-1] -= 10
        t1_2[:,-1] -= 10
        t2_1[:,-1] -= 10
        t2_2[:,-1] -= 10
        t3_1[:,-1] -= 10
        t3_2[:,-1] -= 10
        t4_1[:,-1] -= 10
        t4_2[:,-1] -= 10
        ground_data[:,-1] -= 10
        init_l_data[:,-1] -= 10
        init_r_data[:,-1] -= 10
        # eye_l_data[:,-1] -= 10
        # eye_r_data[:,-1] -= 10

        bias_z = np.mean(ground_data[:,-1]-(init_l_data[:,-1]+init_r_data[:,-1])/2)
        init_l_data[:,2] += bias_z
        init_r_data[:,2] += bias_z

        P1_1 = np.mean(t1_1,axis = 0)
        P1_2 = np.mean(t1_2,axis = 0)
        P1_3 = generate_vector_from_points(P1_1,P1_2,-30.7352,290)
        P1_4 = P1_1+P1_2-P1_3

        P2_1 = np.mean(t2_1,axis = 0)
        P2_2 = np.mean(t2_2,axis = 0)
        P2_3 = generate_vector_from_points(P2_1,P2_2,-38.047,250)
        P2_3[-1] = P2_2[-1]
        P2_4 = P2_1+P2_2-P2_3

        P3_1 = np.mean(t3_1,axis = 0)
        P3_2 = np.mean(t3_2,axis = 0)
        P3_3 = generate_vector_from_points(P3_1,P3_2,30.7352,290)
        P3_4 = P3_1+P3_2-P3_3

        P4_1 = np.mean(t4_1,axis = 0)
        P4_2 = np.mean(t4_2,axis = 0)
        P4_3 = generate_vector_from_points(P4_1,P4_2,-42.614,260)
        P4_3[-1] = P4_2[-1]
        P4_4 = P4_1+P4_2-P4_3

        L_HEEL = df[['l_heel_X', 'l_heel_Y', 'l_heel_Z']].values
        L_TOE = df[['l_toe_X', 'l_toe_Y', 'l_toe_Z']].values
        R_HEEL = df[['r_heel_X', 'r_heel_Y', 'r_heel_Z']].values
        R_TOE = df[['r_toe_X', 'r_toe_Y', 'r_toe_Z']].values

        P =[P1_1, P1_2, P1_3, P1_4, 
            P2_1, P2_2, P2_3, P2_4, 
            P3_1, P3_2, P3_3, P3_4, 
            P4_1, P4_2, P4_3, P4_4]

        P_SAVE = [P1_1,P1_2,P2_1,P2_2,P3_1,ground_data[0]]
        P_SAVE = [P1_1,P1_2,P2_1,P2_2,ground_data[0]]
        # np.save(base_dir + f'/CZK/{sub}_pw_maker.npy',P_SAVE)

        points_src = np.load(base_dir + f'/CZK/{sub}_pw_maker.npy')
        points_dst = np.load(base_dir + f'/CZK/{sub}_pw_tracker.npy')*1e3
        T = estimate_pose_3d3d(points_src, points_dst)
        # print("变换矩阵:\n", T)

        mean_error, individual_errors, transformed_points = calculate_reprojection_error(points_src, points_dst, T)
        print(f"平均重投影误差: {mean_error} mm")
        # print("各点误差:", individual_errors)

        tracker_data = eye_l_data/2+eye_r_data/2
        slam_camera_data = np.load(f'{base_dir}/CZK/250708/saved_traj_{sub}.npy')[:, :3, -1]*1000
        pre_gaze_data = np.load(f'{base_dir}/CZK/{sub}_pre_gaze.npy')*1000
        ab_gaze_data = np.load(f'{base_dir}/CZK/{sub}_ab_gaze.npy')*1000
        gaze_w_all_frame = None
        phases = np.load(f'{base_dir}/CZK/{sub}_phase.npy')

        r = T[:3,:3]
        t = T[:3,-1]
        slam_camera_data = np.matmul(r.T,slam_camera_data.T).T
        t = slam_camera_data[0] - tracker_data[0]
        slam_camera_data -= t
        # gaze_data = np.matmul(r.T,gaze_data.T).T
        # gaze_data -= t 
        gaze_ground_truth = np.array([P1_1/2+P1_2/2,P2_1/2+P2_2/2,P3_1/2+P3_2/2,P4_1/2+P4_2/2])
        pre_errors = []
        ab_errors = []
        if k == 0:
            gaze_w_all_frame = np.load(f'{base_dir}/CZK/{sub}_gaze_all_frame.npy')*1000
            gaze_w_all_frame = np.matmul(r.T,gaze_w_all_frame.T).T -t
        for i,gaze in enumerate(pre_gaze_data):
            
            pre_gaze = np.matmul(r.T,gaze.T).T
            pre_gaze -= t 
            # print(gaze_ground_truth - pre_gaze)
            # print(np.linalg.norm(gaze_ground_truth - pre_gaze,axis=1))
            pre_error = np.sqrt(np.mean(np.linalg.norm(gaze_ground_truth - pre_gaze,axis=1)** 2))
            pre_error = np.mean(np.sqrt((gaze_ground_truth[:,:-1] - pre_gaze[:,:-1])**2))
            pre_errors.append(pre_error)
            pre_gaze_data[i] = pre_gaze

            ab_gaze = np.matmul(r.T,ab_gaze_data[i].T).T
            ab_gaze -= t 
            ab_error = np.mean(np.sqrt((gaze_ground_truth[:,:-1] - ab_gaze[:,:-1])**2))
            # ab_error = np.sqrt(np.mean(np.linalg.norm(gaze_ground_truth - ab_gaze,axis=1)** 2))
            ab_errors.append(ab_error)
            ab_gaze_data[i] = ab_gaze

        # print('Gaze_mean_error:',np.mean(gaze_ground_truth - gaze_data,axis = 0))

        x = np.arange(0,100,5)
        y_ab = np.interp(x, phases, ab_errors).reshape(-1)
        y_map = np.interp(x, phases, pre_errors).reshape(-1)

        err_ab_gaze.append(y_ab)
        err_map_gaze.append(y_map)

        plt.style.use(['science', 'ieee'])  # 使用 'science' 和 'ieee' 样式
        plt.rcParams.update({'figure.dpi': 300})  # 设置图形分辨率为 300 DPI
        plt.rcParams.update({'font.size': 8})   # 设置字体大小为 12

        # plt.figure(figsize=(10,8))
        # plt.plot(phases,np.array(ab_errors)/1000,color = 'b',lw=3)
        # plt.plot(phases,np.array(pre_errors)/1000,'r',lw=3)
        # plt.show()

        len_track_data = len(tracker_data)
        len_camera_data = len(slam_camera_data)
        time_tracker = df['Frame#'].values/120
        time_camera = df['Time'].values[-1]/len_camera_data*np.arange(-1,len(slam_camera_data)-1)
        # print(time_tracker,time_camera)
        distances = cdist(time_camera.reshape(-1, 1), time_tracker.reshape(-1, 1), metric='euclidean')
        down_sample_indices = distances.argmin(axis=1)

        tracker_data = tracker_data[down_sample_indices]
        L_HEEL = L_HEEL[down_sample_indices]
        L_TOE = L_TOE[down_sample_indices]
        R_HEEL = R_HEEL[down_sample_indices]
        R_TOE = R_TOE[down_sample_indices]
        lh_bias = L_HEEL[0,-1]-np.mean(ground_data,axis=0)[-1]
        rh_bias = R_HEEL[0,-1]-np.mean(ground_data,axis=0)[-1]
        lt_bias = L_TOE[0,-1]-np.mean(ground_data,axis=0)[-1]
        rt_bias = R_TOE[0,-1]-np.mean(ground_data,axis=0)[-1]
        L_HEEL[:,-1] -= lh_bias
        L_TOE[:,-1] -= lt_bias
        R_HEEL[:,-1] -= rh_bias
        R_TOE[:,-1] -= rt_bias
        L_FOOT = L_HEEL*0.75+L_TOE*0.25
        R_FOOT = R_HEEL*0.75+R_TOE*0.25
        # L_FOOT = L_HEEL/2+L_TOE/2
        # R_FOOT = R_HEEL/2+R_TOE/2
        time_tracker = time_tracker[down_sample_indices]

        rmse = np.sqrt(np.mean((tracker_data - slam_camera_data)** 2,axis=0))
        ate = np.sqrt(np.mean(np.linalg.norm(tracker_data - slam_camera_data,axis=1)** 2))
        # mse = np.mean((tracker_data - slam_camera_data),axis = 0)
        print(f"ATE: {ate} mm")
        print(f'xyz_rmse:{rmse} mm')
        errors = tracker_data - slam_camera_data
        mean_error = np.mean(errors, axis=0)
        std_error = np.std(errors, axis=0) 
        print('traj_std:',std_error)
        print('\r\n')

        if k ==0:

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlabel('x(mm)', fontsize=8)
            ax.set_ylabel('y(mm)' ,fontsize=8)
            ax.set_zlabel('z(mm)', fontsize=8)
            # ax.set_title('3D Scatter Plot of Markers')

            maker_scatter = ax.scatter([], [], [], c='purple' )
            slam_scatter = ax.scatter([], [], [], c='orange')
            l_heel_scatter = ax.scatter([], [], [], c='pink')
            r_heel_scatter= ax.scatter([], [], [], c='green')
            l_toe_scatter = ax.scatter([], [], [], c='orange')
            r_toe_scatter = ax.scatter([], [], [], c='black')
            l_foot, = ax.plot([], [], [], linestyle='-',c='orange', lw=3)
            r_foot, = ax.plot([], [], [], linestyle='-',c='b', lw=3)
            l_leg, = ax.plot([], [], [], linestyle='-',c='black', lw=3)
            r_leg, = ax.plot([], [], [], linestyle='-',c='black', lw=3)
            hip, = ax.plot([], [], [], linestyle='-',c='black', lw=3)
            body, = ax.plot([], [], [], linestyle='-',c='black', lw=3)
            head, = ax.plot([], [], [], linestyle='-',c='black', lw=3)
            sight, = ax.plot([], [], [], linestyle='-',c='red', lw=3)

            # ax.legend()
            ax.set_xlim([-500, 2000])
            ax.set_ylim([-500, 500])
            ax.set_zlim([0, 1500])
            # ax.set_xticks([0,1000,2000])  # 设置 x 轴刻度位置
            # ax.set_yticks([-250,0,250])  # 设置 y 轴刻度位置
            # ax.set_zticks([0,500,1000,1500])  # 设置 z 轴刻度位置
            verts = [[[-500,-500,-10],[2000,-500,-10],[2000,500,-10],[-500,500,-10]]]  # plane_points 是四个点，连接为一个平面
            poly3d = Poly3DCollection(verts, facecolors='lightskyblue', linewidths=1, edgecolors='black', alpha=0.2)
            ax.add_collection3d(poly3d)
            ax.set_box_aspect([2.5, 1, 1.5])

            P_X = [point[0] for point in P]
            P_Y = [point[1] for point in P]
            P_Z = [point[2] for point in P]

            ax.plot([0, 100], [0, 0], [0, 0], 'r-', linewidth=2)
            ax.plot([0, 0], [0, 100], [0, 0], 'g-', linewidth=2)
            ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2)
            # ax.scatter(P_X,P_Y,P_Z)
            # for gaze in pre_gaze_data:
            #     ax.scatter(gaze[:,0],gaze[:,1],gaze[:,2],color = 'b')
            # for gaze in ab_gaze_data:
            #     ax.scatter(gaze[:,0],gaze[:,1],gaze[:,2],color = 'r') 

            P1 = [[P1_1, P1_3, P1_2, P1_4]]
            P1 = [
                [P1_1, P1_3, P1_2, P1_4],
                [P1_1,P1_3, [*P1_3[:2],-10], [*P1_1[:2],-10]],
                [P1_1,P1_4, [*P1_4[:2],-10], [*P1_1[:2],-10]],
            ]
            P2 = [
                [P2_1, P2_3, P2_2, P2_4],
                [P2_1,P2_3, [*P2_3[:2],-10], [*P2_1[:2],-10]],
                [P2_1,P2_4, [*P2_4[:2],-10], [*P2_1[:2],-10]],
            ]
            P3 = [
                [P3_1, P3_3, P3_2, P3_4],
                [P3_2,P3_4, [*P3_4[:2],-10], [*P3_2[:2],-10]],
                [P3_2,P3_3, [*P3_3[:2],-10], [*P3_2[:2],-10]],
            ]
            P4 = [
                [P4_1, P4_3, P4_2, P4_4],
                [P4_1,P4_3, [*P4_3[:2],-10], [*P4_1[:2],-10]],
                [P4_1,P4_4, [*P4_4[:2],-10], [*P4_1[:2],-10]],
            ]
            # P2 = [[P2_1, P2_3, P2_2, P2_4]]
            # P3 = [[P3_1, P3_3, P3_2, P3_4]]
            # P4 = [[P4_1, P4_3, P4_2, P4_4]]

            draw_plane(P1, ax)
            draw_plane(P2, ax)
            draw_plane(P3, ax)
            draw_plane(P4, ax)

            def update_frame(frame_idx,mode = 'maker'):
                maker_scatter._offsets3d = (tracker_data[:frame_idx, 0], tracker_data[:frame_idx, 1], tracker_data[:frame_idx, 2])
                # l_heel_scatter._offsets3d = (L_HEEL[frame_idx-10:frame_idx, 0], L_HEEL[frame_idx-10:frame_idx, 1], L_HEEL[frame_idx-10:frame_idx, 2])
                # r_heel_scatter._offsets3d = (R_HEEL[frame_idx-10:frame_idx, 0], R_HEEL[frame_idx-10:frame_idx, 1], R_HEEL[frame_idx-10:frame_idx, 2])
                # l_toe_scatter._offsets3d = (L_TOE[frame_idx-10:frame_idx, 0], L_TOE[frame_idx-10:frame_idx, 1], L_TOE[frame_idx-10:frame_idx, 2])
                # r_toe_scatter._offsets3d = (R_TOE[frame_idx-10:frame_idx, 0], R_TOE[frame_idx-10:frame_idx, 1], R_TOE[frame_idx-10:frame_idx, 2])
                slam_scatter._offsets3d = (slam_camera_data[:frame_idx, 0], slam_camera_data[:frame_idx, 1], slam_camera_data[:frame_idx, 2])
                
                marker_traj = np.copy(tracker_data)
                marker_traj[:,0] -=100
                add_foot(ax, L_HEEL, L_TOE, R_HEEL, R_TOE, L_FOOT, R_FOOT, marker_traj,slam_camera_data, gaze_w_all_frame, frame_idx-1, l_foot, r_foot,l_leg,r_leg,hip,body,head,sight)

                plt.draw()
                plt.pause(0.001)

            global poly3d1,poly3d2
            poly3d1 = None
            poly3d2 = None
            legend_elements = [
                    Line2D([0], [0], color='orange', linewidth=3, label='Predict Trajectory'),  # 设置细线
                    Line2D([0], [0], color='purple', linewidth=3, label='Truth Trajectory'),  # 设置细线
                    Line2D([0], [0], color='red', linewidth=3, label='Sight Line')
                ]
            ax.view_init(elev=30, azim=-110)
            ax.legend(handles=legend_elements,loc='upper left',fontsize=10)

            for i in range(1, len(tracker_data)):
                update_frame(i,mode = 'maker')

            if k == 0:
                idx = 96
                print(1)

                ax.plot(slam_camera_data[:idx, 0], slam_camera_data[:idx, 1], slam_camera_data[:idx, 2], linestyle='-', color='darkorange',linewidth=3)#8
                ax.plot(tracker_data[:idx, 0], tracker_data[:idx, 1], tracker_data[:idx, 2],  linestyle='-',color='purple',linewidth=3)
                legend_elements = [
                    Line2D([0], [0], color='orange', linewidth=3, label='Predict Trajectory'),  # 设置细线
                    Line2D([0], [0], color='purple', linewidth=3, label='Truth Trajectory'),  # 设置细线
                    Line2D([0], [0], color='red', linewidth=3, label='Sight Line')
                ]
                # ax.set_xticks([0,1000,2000])  # 设置 x 轴刻度位置
                # ax.set_yticks([-250,0,250])  # 设置 y 轴刻度位置
                # ax.set_zticks([0,500,1000,1500])  # 设置 z 轴刻度位置
                # ax.set_xticklabels([0,1000,2000], fontsize=12)  # 设置 x 轴刻度标签粗体
                # ax.set_yticklabels([-250,0,250], fontsize=12)  # 设置 y 轴刻度标签粗体
                # ax.set_zticklabels([0,500,1000,1500],  fontsize=12)
                # ax.legend(handles=legend_elements,loc='upper left',fontsize=14)
                ax.legend(handles=legend_elements,loc='upper left',fontsize=10)

                marker_traj = np.copy(tracker_data)
                marker_traj[:,0] -=150
                marker_traj[:,2] -=150
                
                # gaze_w_all_frame
                L_HEEL[:,-1] -= 20
                L_TOE[:,-1]-= 20
                R_HEEL[:,-1]-= 20
                R_TOE[:,-1]-= 20
                gaze_w_all_frame[:,1]+= 75
                gaze_w_all_frame[:-1]+= 20
                ax.scatter(gaze_w_all_frame[idx,0],gaze_w_all_frame[idx,1],gaze_w_all_frame[idx,-1],color='b',s=25, marker='*')

                add_foot(ax, L_HEEL, L_TOE, R_HEEL, R_TOE, L_FOOT, R_FOOT, marker_traj,slam_camera_data,gaze_w_all_frame, idx, l_foot, r_foot,l_leg,r_leg,hip,body,head,sight)

                plt.show()

    plt.style.use(['science', 'ieee'])  # 使用 'science' 和 'ieee' 样式
    plt.rcParams.update({'figure.dpi': 300})  # 设置图形分辨率为 300 DPI
    plt.rcParams.update({'font.size': 8}) 

    plt.figure(figsize=(24,8))
    phases = np.arange(0,100,5)
    std_ab = np.std(err_ab_gaze, axis=0)/1000
    std_map = np.std(err_map_gaze, axis=0)/2000
    plt.plot(phases,np.mean(np.array(err_ab_gaze)/1000,axis=0),label="ab_err",linestyle = '-',color = 'b',lw=3)
    plt.fill_between(phases, np.mean(np.array(err_ab_gaze)/1000,axis=0) - std_ab, np.mean(np.array(err_ab_gaze)/1000,axis=0) + std_ab, color='b', alpha=0.1)
    plt.plot(phases,np.mean(np.array(err_map_gaze)/1000,axis=0),label="map_err",linestyle = '-',color ='r',lw=3)
    plt.fill_between(phases, np.mean(np.array(err_map_gaze)/1000,axis=0) - std_map, np.mean(np.array(err_map_gaze)/1000,axis=0) + std_map, color='r', alpha=0.1)
    # plt.title("Basic Plot Example")
    plt.xlabel("Gait Phases(\%)", fontsize=10, color='black')
    plt.ylabel("Gaze Error(m)",fontsize=10, color='black')
    plt.legend(loc='upper left',fontsize=12)
    plt.xticks(fontsize=10)  # 设置 x 轴刻度标签的字体大小
    plt.yticks(fontsize=10)  # 设置 y 轴刻度标签的字体大小
    # plt.legend()
    plt.show()
    
    t_stat, p_value = stats.ttest_ind(np.mean(np.array(err_ab_gaze)/1000,axis=0), np.mean(np.array(err_map_gaze)/1000,axis=0))
    print('ab_error:',np.mean(np.array(err_ab_gaze)/1000))
    print('map_error:',np.mean(np.array(err_map_gaze)/1000))
    print('p_value:',p_value)

def main_field_single():

    sub = 'czk_3'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ground_truth = [[0.7,0.05],[1.30,0.12],[1.20,0.05],[1.15,-0.065],[1.20,-0.05],[1.18,0.02],[1.30,0.025],[1.30,0.00]]
    gait_data = np.load(base_dir + f'/CZK/field_{sub}_gait_param.npy')
    phases = np.load(f'{base_dir}/CZK/field_{sub}_phase.npy')
    
    errors = []
    for each_gait in gait_data:
        print((each_gait-ground_truth)**2)
        error = np.round(np.sqrt(np.mean((100*np.array(each_gait - ground_truth))**2,axis=0)),3)
        # error = np.round(np.mean(100*np.array(each_gait - ground_truth),axis=0),1)
        # error = np.round(100*np.array(each_gait - ground_truth),1)
        # print('Error:',np.round(100*np.array(each_gait - ground_truth),1))
        print('RMSE_error:',error)
        errors.append(error)
    errors = np.array(errors)
    plt.figure(figsize=(10,8))

    plt.subplot(2, 1, 1)
    plt.plot(phases,errors[:,0])

    plt.subplot(2, 1, 2)
    plt.plot(phases,errors[:,1])

    plt.show()

def main_field_all():

    gait_data_all = np.load('data_all.npy')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = []
    x = np.arange(0,100,5)
    phases = np.linspace(0,100,18)
    ground_truth = [[0.6,0.05],[1.30,0.115],[1.15,0.05],[1.25,-0.06],[1.25,-0.05],[1.20,0.02],[1.40,0.025],[1.35,0.00]] #czk
    ground_truth = [[0.55,0.05],[1.30,0.115],[1.25,0.05],[1.30,-0.06],[1.25,-0.05],[1.20,0.02],[1.40,0.00],[1.35,0.00]] #djl删掉最后一步

    global all_peak_bias, all_peak

    data = np.array(gait_data_all['length'])+1
    mean_data_length = []
    for each_data in data:
        mean_data_length.append(np.interp(x, phases, each_data))
    std_data_length = np.std(mean_data_length,axis=0)
    mean_data_length = np.mean(mean_data_length,axis=0)
    print('len_ab:',mean_data_length[all_peak_bias])

    data = np.array(gait_data_all['length_refined'])
    mean_data_length_refined = []
    for each_data in data:
        mean_data_length_refined.append(np.interp(x, phases, each_data))
    std_data_length_refined = np.std(mean_data_length_refined,axis=0)
    mean_data_length_refined = np.mean(mean_data_length_refined,axis=0)
    print('len_map:',mean_data_length_refined[all_peak])

    data = np.array(gait_data_all['height'])
    mean_data_height = []
    for each_data in data:
        mean_data_height.append(np.interp(x, phases, each_data))
    std_data_height = np.std(mean_data_height,axis=0)
    mean_data_height = np.mean(mean_data_height,axis=0)
    print('h_ab:',mean_data_height[all_peak_bias])

    data = np.array(gait_data_all['height_refined'])
    mean_data_height_refined = []
    for each_data in data:
        mean_data_height_refined.append(np.interp(x, phases, each_data))
    std_data_height_refined = np.std(mean_data_height_refined,axis=0)
    mean_data_height_refined = np.mean(mean_data_height_refined,axis=0)
    print('h_map:',mean_data_height_refined[all_peak])

    data = np.array(gait_data_all['slope'])
    mean_data_slope = []
    for each_data in data:
        mean_data_slope.append(np.interp(x, phases, each_data))
    std_data_slope = np.std(mean_data_slope,axis=0)
    mean_data_slope = np.mean(mean_data_slope,axis=0)
    print('slope_ab:',mean_data_slope[all_peak_bias])

    data = np.array(gait_data_all['slope_refined'])
    mean_data_slope_refined = []
    for each_data in data:
        mean_data_slope_refined.append(np.interp(x, phases, each_data))
    std_data_slope_refined = np.std(mean_data_slope_refined,axis=0)
    mean_data_slope_refined = np.mean(mean_data_slope_refined,axis=0)
    print('slope_map:',mean_data_slope_refined[all_peak])

    data = []
    for i in range(5):
        # if i == 2:
        #     continue
        errors = []
        sub = f'djl_{i+1}'
        gait_data = np.load(base_dir + f'/CZK/field_{sub}_gait_param.npy')
        # print(np.mean(gait_data,axis=0))
        for each_gait in gait_data:
            errors.append(each_gait-ground_truth)
            # errors.append(np.sqrt(np.mean((np.array(each_gait - ground_truth))**2,axis=0)))
        phases = np.load(f'{base_dir}/CZK/field_{sub}_phase.npy')

        errors = np.array(errors)
        mean_error = np.mean(errors,axis=1)*100
        mean_error = np.sqrt(np.mean(errors**2,axis=1))*100
        
        # print(mean_error.shape)
        phases = np.load(f'{base_dir}/CZK/field_{sub}_phase.npy')
        y_len = np.interp(x, phases, mean_error[:,0]).reshape(-1)
        y_h = np.interp(x, phases, mean_error[:,1]).reshape(-1)
        data.append(np.vstack((y_len,y_h)).T)
    print(np.array(data).shape)
    std = np.std(data,axis=0)
    data = np.mean(data,axis=0)
    mean_bias_djl = []
    std_bias_djl = []

    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({'figure.dpi': 300})
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(10,8))
    plt.figtext(0.5, 0.07, 'Gait phase during swing(\%)', ha='center', va='center', fontsize=8)

    plt.subplot(1, 3, 1)
    # plt.plot(x,data[:,0],linestyle = '-',color = 'steelblue',lw=2)
    # plt.fill_between(x, data[:,0] - std[:,0],  data[:,0] + std[:,0], color='b', alpha=0.1)
    plt.plot(x,mean_data_length,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_length - std_data_length,  mean_data_length + std_data_length, color='darkorange', alpha=0.1)
    plt.plot(x,mean_data_length_refined,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_length_refined - std_data_length_refined,  mean_data_length_refined + std_data_length_refined, color='steelblue', alpha=0.1)
    plt.ylabel("Step length error(cm)", fontsize=7, color='black')
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    # plt.xlabel("Gait phase during swing(\%)", fontsize=10, color='black')
    # plt.xlim([0,102])

    plt.subplot(1, 3, 2)
    plt.plot(x,mean_data_height,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_height - std_data_height,  mean_data_height + std_data_height, color='darkorange', alpha=0.1)
    plt.plot(x,mean_data_height_refined,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_height_refined - std_data_height_refined,  mean_data_height_refined + std_data_height_refined, color='steelblue', alpha=0.1)
    plt.ylabel("Step height error(cm)", fontsize=7, color='black')
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    # plt.xlabel("Gait phase during swing(\%)", fontsize=10, color='black')

    plt.subplot(1, 3, 3)
    plt.plot(x,mean_data_slope,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_slope - std_data_slope,  mean_data_slope + std_data_slope, color='darkorange', alpha=0.1)
    plt.plot(x,mean_data_slope_refined,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_slope_refined - std_data_slope_refined,  mean_data_slope_refined + std_data_slope_refined, color='steelblue', alpha=0.1)
    plt.ylabel("footholf slope error(${}^\circ$)", fontsize=7, color='black')
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    plt.ylim([1.8,4.8])
    # plt.xlabel("Gait phase during swing(\%)", fontsize=10, color='black')

    lines = [
            Line2D([0], [0], color='darkorange', linewidth=2.5, label='Estimated by Single Gaze'),  # 设置细线
            Line2D([0], [0], color='steelblue', linewidth=2.5, label='Estimated by Attention Map'),]
    fig.legend(
        handles=lines,
        loc='upper center',  # 图例位置
        fontsize=7,  # 字体大小
        bbox_to_anchor=(0.5, 1.07),  # 图例位置调整到顶部
        ncol=2  # 设置图例项的列数
    )
    plt.show()


    length_data_mean = np.array([[12.3,15.6,11.2,10.4,14.3],
                        [6.4,9.1,5.6,5.5,7.7]]).T
    
    length_data_std = np.array([[3.3,5.6,4.2,5.4,6.3],
                        [2.4,3.1,3.6,4.5,4.3]]).T

    height_data_mean = np.array([[3.05,3.68,2.64,2.74,3.38],
                                 [1.71,2.64,1.84,1.65,1.74]]).T
    
    height_data_std = np.array([[1.11,1.25,1.24,1.37,1.75],
                                [0.85,0.82,0.61,0.95,1.20]]).T

    slope_data_mean = np.array([[2.55,4.64,3.13,2.64,4.05],
                                [2.11,3.65,2.74,2.15,2.01]]).T
    
    slope_data_std = np.array([[0.95,1.15,1.24,1.07,1.35],
                               [0.65,0.72,0.91,0.65,1.10]]).T

    fig = plt.figure(figsize=(10,8))
    subjects = ['1', '2', '3', '4', '5']
    conditions = ['Estimated by Single Gaze', 'Estimated by Attention Map']
    colors = ['green','firebrick']
    width = 0.4
    x = np.arange(len(subjects))

    plt.rcParams.update({'font.size': 9})

    plt.subplot(1, 3, 1)
    for i, condition in enumerate(conditions):
        plt.bar(x + i * width, length_data_mean[:, i], width, yerr=length_data_std[:, i]
                , capsize=1, label=condition, color=colors[i], edgecolor='black', linewidth=0.8)
    plt.ylim([0,25])
    plt.xticks(x + width/2, subjects)
    plt.xlabel('Subjects')
    plt.ylabel('Error of predicted step length (cm)',fontsize=8)

    plt.subplot(1, 3, 2)
    for i, condition in enumerate(conditions):
        plt.bar(x + i * width, height_data_mean[:, i], width, yerr=height_data_std[:, i]
                , capsize=1, color=colors[i], edgecolor='black', linewidth=0.8)
    plt.ylim([0,6])
    plt.xticks(x + width/2, subjects)
    plt.xlabel('Subjects')
    plt.ylabel('Error of predicted step height (cm)',fontsize=8)

    plt.subplot(1, 3, 3)
    for i, condition in enumerate(conditions):
        plt.bar(x + i * width, slope_data_mean[:, i], width, yerr=slope_data_std[:, i]
                , capsize=1, color=colors[i], edgecolor='black', linewidth=0.8)
    plt.ylim([0,7])
    plt.xticks(x + width/2, subjects)
    plt.xlabel('Subjects')
    plt.ylabel('Error of predicted slope (°)',fontsize=8)

    fig.legend(loc='upper center',fontsize=7, bbox_to_anchor=(0.5, 1.02), ncol=2)
    # plt.figtext(0.5, 0.04, 'Subjects', ha='center', va='center', fontsize=8)
    plt.show()
    import math

    def calculate_two_sample_p_value(mean1, mean2, std1, std2, n1, n2):
        # 计算 t 统计量
        t_stat = (mean1 - mean2) / math.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        # 计算自由度
        df = ((std1**2 / n1 + std2**2 / n2)**2) / ((std1**4 / (n1**2 * (n1 - 1))) + (std2**4 / (n2**2 * (n2 - 1))))
        
        # 计算双尾 P 值
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        
        return p_value
    
    p_len = calculate_two_sample_p_value(np.mean(length_data_mean,axis=0)[0], np.mean(length_data_mean,axis=0)[-1],
                                          np.mean(length_data_std,axis=0)[0], np.mean(length_data_std,axis=0)[-1], 12, 12)
    p_len = calculate_two_sample_p_value(np.mean(height_data_mean,axis=0)[0], np.mean(height_data_mean,axis=0)[-1],
                                          np.mean(height_data_std,axis=0)[0], np.mean(height_data_std,axis=0)[-1], 12, 12)
    # p_len = calculate_two_sample_p_value(np.mean(slope_data_mean,axis=0)[0], np.mean(slope_data_mean,axis=0)[-1],
    #                                       np.mean(slope_data_std,axis=0)[0], np.mean(slope_data_std,axis=0)[-1], 12, 12)
    
    print(p_len)

def main_field_acc():

    acc = np.load('data_acc.npy')
    
    acc_bias = np.load('data_acc_bias.npy')

    x = np.arange(0,100,5)
    x_ = np.arange(0,100,1)
    phases = np.linspace(0,100,18)
    
    data = np.array(acc['czk'])*100/8
    data_czk = []
    for each_data in data:
        y_czk = np.interp(x, phases, each_data)
        data_czk.append(y_czk)
    std_czk = np.std(data_czk,axis=0)
    mean_data_czk = np.mean(data_czk,axis = 0)
    czk_start, czk_end,czk_start_idx, czk_end_idx, mean_data_czk_100 = find_window_above(mean_data_czk)
    czk_peak = detect_peaks(mean_data_czk,mph = 90,mpd=20)

    data = np.array(acc_bias['czk'])*100/8
    data_czk_bias = []
    for each_data in data:
        y_czk = np.interp(x, phases, each_data)
        data_czk_bias.append(y_czk)
    std_czk_bias = np.std(data_czk_bias,axis=0)
    mean_data_czk_bias= np.mean(data_czk_bias,axis = 0)
    czk_start_bias, czk_end_bias,czk_start_idx_bias, czk_end_idx_bias, mean_data_czk_bias_100 = find_window_above(mean_data_czk_bias)
    czk_peak_bias = detect_peaks(mean_data_czk_bias,mph = 90,mpd=20)

    data = np.array(acc['djl'])*100/8
    data_djl = []
    for each_data in data:
        y_djl = np.interp(x, phases, each_data)
        data_djl.append(y_djl)
    std_djl = np.std(data_djl,axis=0)
    # print(std_djl[-1])
    mean_data_djl = np.mean(data_djl,axis = 0)
    djl_start, djl_end,djl_start_idx, djl_end_idx, mean_data_djl_100 = find_window_above(mean_data_djl)
    djl_peak = detect_peaks(mean_data_djl,mph = 90,mpd=20)

    data = np.array(acc_bias['djl'])*100/8
    data_djl_bias = []
    for each_data in data:
        y_djl = np.interp(x, phases, each_data)
        data_djl_bias.append(y_djl)
    std_djl_bias = np.std(data_djl_bias,axis=0)
    # print(std_djl_bias[-1])
    mean_data_djl_bias = np.mean(data_djl_bias,axis = 0)
    djl_start_bias, djl_end_bias,djl_start_idx_bias, djl_end_idx_bias, mean_data_djl_bias_100 = find_window_above(mean_data_djl_bias)
    djl_peak_bias = detect_peaks(mean_data_djl_bias,mph = 90,mpd=20)

    data = np.array(acc['wwh'])*100/8
    data_wwh = []
    for each_data in data:
        y_wwh = np.interp(x, phases, each_data)
        data_wwh.append(y_wwh)
    std_wwh = np.std(data_wwh,axis=0)
    mean_data_wwh = np.mean(data_wwh,axis = 0)
    wwh_start, wwh_end, wwh_start_idx, wwh_end_idx, mean_data_wwh_100 = find_window_above(mean_data_wwh)
    wwh_peak = detect_peaks(mean_data_wwh,mph = 90,mpd=20)

    data = np.array(acc_bias['wwh'])*100/8
    data_wwh_bias = []
    for each_data in data:
        y_wwh = np.interp(x, phases, each_data)
        data_wwh_bias.append(y_wwh)
    std_wwh_bias = np.std(data_wwh_bias,axis=0)
    mean_data_wwh_bias = np.mean(data_wwh_bias,axis = 0)
    wwh_start_bias, wwh_end_bias,wwh_start_idx_bias, wwh_end_idx_bias, mean_data_wwh_bias_100 = find_window_above(mean_data_wwh_bias)
    wwh_peak_bias = detect_peaks(mean_data_wwh_bias,mph = 90,mpd=20)

    data = np.array(acc['hlh'])*100/8
    data_hlh = []
    for each_data in data:
        y_hlh = np.interp(x, phases, each_data)
        data_hlh.append(y_hlh)
    std_hlh = np.std(data_hlh,axis=0)
    mean_data_hlh = np.mean(data_hlh,axis = 0)
    hlh_start, hlh_end, hlh_start_idx, hlh_end_idx, mean_data_hlh_100 = find_window_above(mean_data_hlh)
    hlh_peak = detect_peaks(mean_data_hlh,mph = 90,mpd=20)

    data = np.array(acc_bias['hlh'])*100/8
    data_hlh_bias = []
    for each_data in data:
        y_hlh = np.interp(x, phases, each_data)
        data_hlh_bias.append(y_hlh)
    std_hlh_bias = np.std(data_hlh_bias,axis=0)
    mean_data_hlh_bias = np.mean(data_hlh_bias,axis = 0)
    hlh_start_bias, hlh_end_bias,hlh_start_idx_bias, hlh_end_idx_bias, mean_data_hlh_bias_100 = find_window_above(mean_data_hlh_bias)
    hlh_peak_bias = detect_peaks(mean_data_hlh_bias,mph = 90,mpd=20)

    data = np.array(acc['xhy'])*100/8
    data_xhy = []
    for each_data in data:
        y_xhy = np.interp(x, phases, each_data)
        data_xhy.append(y_xhy)
    std_xhy = np.std(data_xhy,axis=0)
    mean_data_xhy = np.mean(data_xhy,axis = 0)
    xhy_start, xhy_end,xhy_start_idx, xhy_end_idx, mean_data_xhy_100 = find_window_above(mean_data_xhy)
    xhy_peak = detect_peaks(mean_data_xhy,mph = 90,mpd=20)

    data = np.array(acc_bias['xhy'])*100/8
    data_xhy_bias = []
    for each_data in data:
        y_xhy = np.interp(x, phases, each_data)
        data_xhy_bias.append(y_xhy)
    std_xhy_bias = np.std(data_xhy_bias,axis=0)
    mean_data_xhy_bias = np.mean(data_xhy_bias,axis = 0)
    xhy_start_bias, xhy_end_bias,xhy_start_idx_bias, xhy_end_idx_bias, mean_data_xhy_bias_100 = find_window_above(mean_data_xhy_bias)
    xhy_peak_bias = detect_peaks(mean_data_xhy_bias,mph = 90,mpd=20)

    global all_peak_bias, all_peak

    all_data = np.array([data_czk,data_djl,data_hlh,data_wwh,data_xhy]).reshape(-1,20)
    std_all = np.std(all_data,axis=0)
    mean_data_all = np.mean(all_data,axis = 0)
    all_start, all_end,all_start_idx, all_end_idx, mean_data_all_100 = find_window_above(mean_data_all)
    all_peak = detect_peaks(mean_data_all,mph = 90,mpd=20)
    print('all_peak:',phases[all_peak],mean_data_all[all_peak],phases[all_end_idx] - phases[all_start_idx])

    all_data_bias = np.array([data_czk_bias,data_djl_bias,data_hlh_bias,data_wwh_bias,data_xhy_bias]).reshape(-1,20)
    std_all_bias = np.std(all_data_bias,axis=0)
    mean_data_all_bias = np.mean(all_data_bias,axis = 0)
    all_start_bias, all_end_bias,all_start_idx_bias, all_end_idx_bias, mean_data_all_bias_100 = find_window_above(mean_data_all_bias)
    all_peak_bias = detect_peaks(mean_data_all_bias,mph = 90,mpd=20)
    print('all_peak_bias:',phases[all_peak_bias],mean_data_all_bias[all_peak_bias],phases[all_end_idx_bias] - phases[all_start_idx_bias])

    np.save('all_accuracy.npy',mean_data_all)
    np.save('all_accuracy_bias.npy',mean_data_all_bias)

    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({'figure.dpi': 300})
    plt.rcParams.update({'font.size': 7})

    fig=plt.figure(figsize=(8,2))
    plt.figtext(0.5, 0.06, 'Gait phase during swing (\%)', ha='center', va='center', fontsize=8)

    plt.subplot(1, 6, 1)
    plt.plot(x,mean_data_czk,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_czk - std_czk, mean_data_czk + std_czk, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_czk_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_czk_bias - std_czk_bias, mean_data_czk_bias + std_czk_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[czk_start],x_[czk_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[czk_start_bias],x_[czk_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[czk_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[czk_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    # plt.gca().set_yticklabels([])
    plt.title("Subject 1")
    plt.ylabel("Prdict Accuracy (\%)", fontsize=9, color='black')
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    # plt.xlabel("subject 1",fontsize=8, color='black')
    
    plt.subplot(1, 6, 2)
    plt.plot(x,mean_data_djl,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_djl - std_djl, mean_data_djl + std_djl, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_djl_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_djl_bias - std_djl_bias, mean_data_djl_bias+ std_djl_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[djl_start],x_[djl_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[djl_start_bias],x_[djl_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[djl_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[djl_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    plt.title("Subject 2")
    plt.gca().set_yticklabels([])
    plt.xticks(fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    # plt.xlabel("subject 2",fontsize=8, color='black')
    
    plt.subplot(1, 6, 3)
    plt.plot(x,mean_data_wwh,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_wwh - std_wwh, mean_data_wwh + std_wwh, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_wwh_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_wwh_bias - std_wwh_bias, mean_data_wwh_bias+ std_wwh_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[wwh_start],x_[wwh_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[wwh_start_bias],x_[wwh_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[wwh_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[wwh_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    plt.title("Subject 3")
    plt.gca().set_yticklabels([])
    plt.xticks(fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度
    # plt.xlabel("subject 3",fontsize=8, color='black')

    plt.subplot(1, 6, 4)
    plt.plot(x,mean_data_hlh,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_hlh - std_hlh, mean_data_hlh + std_hlh, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_hlh_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_hlh_bias - std_hlh_bias, mean_data_hlh_bias + std_hlh_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[hlh_start],x_[hlh_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[hlh_start_bias],x_[hlh_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[hlh_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[hlh_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    plt.gca().set_yticklabels([])
    plt.title("Subject 4")
    plt.xticks(fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度

    plt.subplot(1, 6, 5)
    plt.plot(x,mean_data_xhy,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_xhy - std_xhy, mean_data_xhy + std_xhy, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_xhy_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_xhy_bias - std_xhy_bias, mean_data_xhy_bias + std_xhy_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[xhy_start],x_[xhy_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[xhy_start_bias],x_[xhy_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[xhy_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[xhy_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    plt.title("Subject 5")
    plt.xticks(fontsize=9)
    plt.gca().set_yticklabels([])
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度

    plt.subplot(1, 6, 6)
    plt.plot(x,mean_data_all,linestyle = '-',color = 'steelblue',lw=2)
    plt.fill_between(x, mean_data_all - std_all, mean_data_all + std_all, color='steelblue', alpha=0.15)
    plt.plot(x,mean_data_all_bias,linestyle = '--',color = 'darkorange',lw=2)
    plt.fill_between(x, mean_data_all_bias - std_all_bias, mean_data_all_bias + std_all_bias, color='darkorange', alpha=0.15)
    plt.plot([x_[all_start],x_[all_end]],[55,55],linestyle = '-',color = 'purple',lw=2)
    plt.plot([x_[all_start_bias],x_[all_end_bias]],[57.5,57.5],linestyle = '-',color = 'green',lw=2)
    plt.scatter(x[all_peak],[55],s=10,color = 'firebrick',zorder=3)
    plt.scatter(x[all_peak_bias],[57.5],s=10,color = 'firebrick',zorder=3)
    plt.ylim([50,102])
    plt.gca().set_yticklabels([])
    plt.xticks(fontsize=9)
    plt.title("Mean of All")
    plt.tick_params(axis='x', which='minor', length=0)  # 去除 x 轴的小刻度
    plt.tick_params(axis='y', which='minor', length=0)  # 去除 y 轴的小刻度

    lines = [
            Line2D([0], [0], color='darkorange', linewidth=2, label='Estimated by Single Gaze'),  # 设置细线
            Line2D([0], [0], color='steelblue', linewidth=2, label='Estimated by Attention Map'),  # 设置细线
            Line2D([0], [0], color='green', linewidth=2, label='${W}_{p(90)}$ of Single Gaze'),
            Line2D([0], [0], color='purple', linewidth=2, label='${W}_{p(90)}$ of Attention Map'),
            Line2D([0], [0], color='firebrick', marker='o', markersize=2,linewidth=0, label='Best Prediction Phase')]
    fig.legend(
        handles=lines,
        loc='upper center',  # 图例位置
        fontsize=7,  # 字体大小
        bbox_to_anchor=(0.5, 1.02),  # 图例位置调整到顶部
        ncol=5  # 设置图例项的列数
    )
    plt.show()

    fig=plt.figure(figsize=(24,6))
    plt.rcParams.update({'font.size': 9})

    data = [mean_data_czk[czk_start_idx:czk_end_idx+1],mean_data_czk_bias[czk_start_idx_bias:czk_end_idx_bias+1],
            mean_data_djl[djl_start_idx:djl_end_idx+1],mean_data_djl_bias[djl_start_idx_bias:djl_end_idx_bias+1],
            mean_data_wwh[wwh_start_idx:wwh_end_idx+1],mean_data_wwh_bias[wwh_start_idx_bias:wwh_end_idx_bias+1],
            mean_data_hlh[hlh_start_idx:hlh_end_idx+1],mean_data_hlh_bias[hlh_start_idx_bias:hlh_end_idx_bias+1],
            mean_data_xhy[xhy_start_idx:wwh_end_idx+1],mean_data_xhy_bias[xhy_start_idx_bias:xhy_end_idx_bias+1],
            mean_data_all[all_start_idx:all_end_idx+1],mean_data_all_bias[all_start_idx_bias:all_end_idx_bias+1]]
    
    # data = [mean_data_czk_100[czk_start:czk_end+1],mean_data_czk_bias_100[czk_start_bias:czk_end_bias+1],
    #         mean_data_djl_100[djl_start:djl_end+1],mean_data_djl_bias_100[djl_start_bias:djl_end_bias+1],
    #         mean_data_wwh_100[wwh_start:wwh_end+1],mean_data_wwh_bias_100[wwh_start_bias:wwh_end_bias+1],
    #         mean_data_hlh_100[hlh_start:hlh_end+1],mean_data_hlh_bias_100[hlh_start_bias:hlh_end_bias+1],
    #         mean_data_xhy_100[xhy_start:wwh_end+1],mean_data_xhy_bias_100[xhy_start_bias:xhy_end_bias+1],
    #         mean_data_all_100[wwh_start:wwh_end+1],mean_data_all_bias_100[all_start_bias:all_end_bias+1]]
    
    # print(data)

    box = plt.boxplot(data, positions=[0,1,2.5,3.5,5,6,7.5,8.5,10,11,12.5,13.5], widths=0.6,patch_artist=True)
    flag = 1
    for i,patch in enumerate(box['boxes']):
        if i%2 ==0:
            patch.set_facecolor('lightblue')  # 设置箱体的填充颜色
            if flag:
                patch.set_label('Estimated by Attention Map in ${W}_{p(90)}$')
                flag +=1
        else:
            patch.set_facecolor('lightgreen')  # 设置箱体的填充颜色
            if flag:
                patch.set_label('Estimated by Single Gaze in ${W}_{p(90)}$')
                flag+=1
        if flag == 3:
            flag = 0
        patch.set_edgecolor('black')  # 设置箱体边框颜色

    # 设置中位线颜色（中位线是 Line2D 类型）
    for median in box['medians']:
        median.set_color('red')  # 设置中位线的颜色

    # 设置胡须和顶端、底端的颜色（胡须和顶端、底端是 Line2D 类型）
    for whisker in box['whiskers']:
        whisker.set_color('black')  # 设置胡须的颜色
    for cap in box['caps']:
        cap.set_color('black')  # 设置顶端和底端的颜色

    plt.xticks([0.5,3,5.5,8,10.5,13], ['subject 1', 'subject 2', 'subject 3', 'subject 4', 'subject 5', 'Mean of All'],fontsize=8)
    plt.ylabel("Prdict Accuracy (\%)", fontsize=9, color='black')
    fig.legend(loc='upper center',  # 图例位置
        fontsize=6,  # 字体大小
        bbox_to_anchor=(0.5, 1.03),  # 图例位置调整到顶部
        ncol=2)
    plt.grid(True)
    # plt.figtext(0.5, 0.03, 'Subjects', ha='center', va='center', fontsize=10)
    plt.show()
    ab_means = np.array([np.mean(item) for item in data[1::2]])
    ab_overall_mean = np.mean(ab_means)
    map_means = np.array([np.mean(item) for item in data[0::2]])
    map_overall_mean = np.mean(map_means)
    print('mean_Wp90_ab:',ab_overall_mean)
    print('mean_Wp90_map:',map_overall_mean)
    # ab_data = [item for item in data[1::2]]
    # map_data = [item for item in data[0::2]]
    import scipy.stats as stats
    t_stat, p_value = stats.ttest_ind(mean_data_all[all_start_idx:all_end_idx+1], mean_data_all_bias[all_start_idx_bias:all_end_idx_bias+1])
    print('90_p:',p_value)
    t_stat, p_value = stats.ttest_ind(mean_data_all, mean_data_all_bias)
    print('all_swing_p:',p_value)

def imu_plot():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    imu_data = np.load(base_dir+'/CZK/saved_imu_data.npy')[110:334]

    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({'figure.dpi': 300})
    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(imu_data,linestyle = '-',color = 'steelblue', lw=1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 画带箭头的坐标轴（使用轴坐标系，0~1）
    # x 轴箭头
    ax.annotate(
        '', xy=(1.05, 0.0), xytext=(-0.008, 0.0),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1, mutation_scale=6), zorder=10
    )
    # y 轴箭头
    ax.annotate(
        '', xy=(0.0, 1.05), xytext=(0.0, -0.008),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', lw=1, mutation_scale=6), zorder=10
    )

    # ax.text(0.4, -0.04, 'Time', transform=ax.transAxes, ha='left', va='top',fontsize=5)
    # ax.text(-0.04, 0.5, r'$\omega_x$', transform=ax.transAxes, ha='right', va='bottom',fontsize=6, rotation=90)
    ax.text(0.4, -0.04, 'Time', transform=ax.transAxes, ha='left', va='top',fontsize=7)
    ax.text(-0.04, 0.5, r'$\omega_x$', transform=ax.transAxes, ha='right', va='bottom',fontsize=8, rotation=90)

    plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12)

    plt.show()

def imu_video():
    dt_l = np.load('/home/czk/left_heel.npy')[::2]
    dt_r = np.load('/home/czk/right_heel.npy')[::2]
    lto  = np.load('/home/czk/lto.npy')[1::2] + np.load('/home/czk/lto.npy')[::2]
    lto[lto!=2000] -= 1000 
    rto  = np.load('/home/czk/rto.npy')[1::2] + np.load('/home/czk/rto.npy')[::2]
    rto[rto!=2000] -= 1000
    lhs  = np.load('/home/czk/lhs.npy')[1::2] + np.load('/home/czk/lhs.npy')[::2]
    lhs[lhs!=2000] -= 1000
    rhs  = np.load('/home/czk/rhs.npy')[1::2] + np.load('/home/czk/rhs.npy')[::2]
    rhs[rhs!=2000] -= 1000
    dto = np.copy(lto)
    dto[rto!=2000] = rto[rto!=2000]
    dhs = np.copy(lhs)
    dhs[rhs!=2000] = rhs[rhs!=2000]
    print(len(dt_l))

    t = np.arange(len(dt_l)) * 9.3 / len(dt_l)

    # 可选：SciencePlots 风格包可能没装，做个保护
    # try:
    #     plt.style.use(['science', 'ieee'])
    # except Exception:
    #     pass

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(24, 6))

    # 先固定坐标轴范围
    ax.set_ylim(-275, 275)
    ax.set_xlim(-0.5, 10)
    ax.set_xlabel('Time (s)', fontsize=16, labelpad=6)      # X 轴标签
    ax.set_ylabel('Angular velocity (°/s)', fontsize=16, labelpad=6)   # Y 轴标签

    # 预创建两条空线（颜色与你原代码一致）
    line_l, = ax.plot([], [], '-', color='darkorange', lw=4)
    line_r, = ax.plot([], [], '-', color='steelblue',  lw=4)
    to = ax.scatter([], [], marker='o',facecolors='none',
                    edgecolors='red',s=60, linewidths=1.5,zorder=3)
    hs = ax.scatter([], [], marker='s',facecolors='none',
                    edgecolors='black',s=60, linewidths=1.5,zorder=3)
    # ax.legend(loc='upper right')
    ms_to = np.sqrt(60)   # s=60 -> markersize≈sqrt(60)
    ms_hs = np.sqrt(60)
    from matplotlib.patches import Rectangle

    def make_span_patches(ax, intervals, color='#82c8a0', alpha=0.25):
        """为每段区间创建一个竖直条带（满高），初始 width=0。"""
        patches = []
        for i,(start, end) in enumerate(intervals):
            if i%2 ==0:
                color = '#82c8a0'
            else:
                color = 'orange'
            r = Rectangle(
                (start, 0),      # 左下角 (x=start, y=0)
                0.0,             # 初始宽度为 0（后续更新）
                1.0,             # 高度=1（配合 xaxis_transform -> 跨满 y 轴）
                transform=ax.get_xaxis_transform(),
                facecolor=color, edgecolor='none', alpha=alpha, zorder=0.2
            )
            ax.add_patch(r)
            patches.append(r)
        return patches

    def update_span_patches(patches, intervals, cur_t):
        for r, (start, end) in zip(patches, intervals):
            if cur_t <= start:
                r.set_x(start); r.set_width(0.0)
            elif cur_t >= end:
                r.set_x(start); r.set_width(end - start)
            else:
                r.set_x(start); r.set_width(cur_t - start)

    intervals = []
    to_ids = t[dto!=2000]
    hs_ids = t[dhs!=2000]
    for i in range(len(to_ids)):
        intervals.append((to_ids[i],hs_ids[i]))

    patches = make_span_patches(ax, intervals)
    N = len(t)

    lines = [
                Line2D([0], [0], color='darkorange', linewidth=2.5, label='Left Heel'),  # 设置细线
                Line2D([0], [0], color='steelblue', linewidth=2.5, label='Right Heel'),
                Line2D([0], [0], linestyle='None', marker='o',
                    markerfacecolor='none', markeredgecolor='red',markeredgewidth=1.8,
                    markersize=ms_to, label='Toe-off'),
                Line2D([0], [0], linestyle='None', marker='s',
                    markerfacecolor='none', markeredgecolor='black',markeredgewidth=1.8,
                    markersize=ms_hs, label='Heel-strike'),
                Patch(facecolor='#82c8a0', alpha=0.5, edgecolor='none', label='Left band'),
                Patch(facecolor='orange', alpha=0.5, edgecolor='none', label='Left band')]
    fig.legend(
            handles=lines,
            loc='upper center',  # 图例位置
            fontsize=16,  # 字体大小
            bbox_to_anchor=(0.5, 1.0),  # 图例位置调整到顶部
            ncol=6  # 设置图例项的列数
        )

    N = len(t)
    interval_ms = 1000 * 9.3 / N   # 一帧对应的物理时间（毫秒）

    idx = {'i': 0}                 # 用可变对象在闭包里保存状态

    def step():
        i = idx['i']
        if i >= N:
            timer.stop()           # 播放完停止定时器
            return

        cur_t = t[i]

        # —— 你的原更新逻辑 —— #
        update_span_patches(patches, intervals, cur_t)

        m_to = (dto != 2000) & (t <= cur_t)
        m_hs = (dhs != 2000) & (t <= cur_t)
        to.set_offsets(np.c_[t[m_to],  dto[m_to]] if np.any(m_to) else np.empty((0,2)))
        hs.set_offsets(np.c_[t[m_hs],  dhs[m_hs]] if np.any(m_hs) else np.empty((0,2)))

        line_l.set_data(t[:i+1], dt_l[:i+1])
        line_r.set_data(t[:i+1], dt_r[:i+1])

        fig.canvas.draw_idle()     # 触发重绘（由 GUI 事件循环调度）

        idx['i'] = i + 1

    # 创建并启动定时器
    timer = fig.canvas.new_timer(interval=interval_ms)
    timer.add_callback(step)
    timer.start()

    plt.show()

    
if __name__ == '__main__':

    # main_maker()
    # main_field_single()
    # main_field_acc()
    # main_field_all()
    # imu_plot()
    imu_video()

    # plt.figure(figsize=(10,8))
    # plt.show()