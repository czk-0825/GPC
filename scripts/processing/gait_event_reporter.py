import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter1d
from detecta import detect_peaks
from scipy.spatial.distance import cdist
import yaml
import numpy as np

address = '/home/czk/catkin_ws/src/samcon_perception/scripts/GPP/CZK/250708/'
sub = 'czk_1'
df = pd.read_csv(f'{address}Trimmed_exp_{sub}.trc', delim_whitespace=True, skiprows=5, header=None)  # 跳过头部

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
bias_z = np.mean(ground_data[:,-1]-(init_l_data[:,-1]+init_r_data[:,-1])/2)
init_l_data[:,2] += bias_z
init_r_data[:,2] += bias_z


tracker_data = eye_l_data/2+eye_r_data/2
slam_camera_data = np.load(f'{address}saved_traj_{sub}.npy')[:, :3, -1]*1000
slam_camera_data[:, [0, 1]] = slam_camera_data[:, [1, 0]]
slam_camera_data[:, 1] = -slam_camera_data[:, 1]
slam_camera_data -= slam_camera_data[0]
slam_camera_data += tracker_data[0]
len_track_data = len(tracker_data)
len_camera_data = len(slam_camera_data)
time_tracker = df['Frame#'].values/120
time_camera = df['Time'].values[-1]/len_camera_data*np.arange(1,len(slam_camera_data)+1)
distances = cdist(time_camera.reshape(-1, 1), time_tracker.reshape(-1, 1), metric='euclidean')
down_sample_indices = distances.argmin(axis=1)
tracker_data = tracker_data[down_sample_indices]
time_tracker = time_tracker[down_sample_indices]


# 提取左脚跟和右脚跟的坐标（单位转换：mm -> m）
l_heel_x = df['l_heel_X'].values * 1e-3
l_heel_y = df['l_heel_Y'].values * 1e-3
l_heel_z = df['l_heel_Z'].values * 1e-3

r_heel_x = df['r_heel_X'].values * 1e-3
r_heel_y = df['r_heel_Y'].values * 1e-3
r_heel_z = df['r_heel_Z'].values * 1e-3

# 获取时间列
# time = df['Time'].values
time = df['Frame#'].values/120

# 计算左脚跟的位置变化（速度）
l_heel_velocity_x = np.diff(l_heel_x) / 0.008333
l_heel_velocity_y = np.diff(l_heel_y) / 0.008333
l_heel_velocity_z = np.diff(l_heel_z) / 0.008333

# 计算右脚跟的位置变化（速度）
r_heel_velocity_x = np.diff(r_heel_x) / 0.008333
r_heel_velocity_y = np.diff(r_heel_y) / 0.008333
r_heel_velocity_z = np.diff(r_heel_z) / 0.008333

# 计算左脚跟的加速度（加速度 = 速度变化 / 时间差）
l_heel_acceleration_x = np.diff(l_heel_velocity_x) / 0.008333
l_heel_acceleration_y = np.diff(l_heel_velocity_y) / 0.008333
l_heel_acceleration_z = np.diff(l_heel_velocity_z) / 0.008333

# 计算右脚跟的加速度（加速度 = 速度变化 / 时间差）
r_heel_acceleration_x = np.diff(r_heel_velocity_x) / 0.008333
r_heel_acceleration_y = np.diff(r_heel_velocity_y) / 0.008333
r_heel_acceleration_z = np.diff(r_heel_velocity_z) / 0.008333

# 计算左脚跟和右脚跟的总加速度
# total_l_heel_v = np.sqrt(
#     l_heel_velocity_x**2 + l_heel_velocity_y**2 + l_heel_velocity_z**2
# )

# total_r_heel_v = np.sqrt(
#     r_heel_velocity_x**2 + r_heel_velocity_y**2 + r_heel_velocity_z**2
# )

# total_l_heel_acceleration = np.sqrt(
#     l_heel_acceleration_x**2 + l_heel_acceleration_y**2 + l_heel_acceleration_z**2
# )

# total_r_heel_acceleration = np.sqrt(
#     r_heel_acceleration_x**2 + r_heel_acceleration_y**2 + r_heel_acceleration_z**2
# )

total_l_heel_v = l_heel_velocity_x

total_r_heel_v = r_heel_velocity_x

total_l_heel_acceleration = l_heel_acceleration_x

total_r_heel_acceleration = r_heel_acceleration_x

# 创建一个新的DataFrame来存储加速度数据
heel_acceleration_df = pd.DataFrame({
    'Time': time[2:],  # 总加速度的时间从第3个点开始
    'Total_L_Heel_Acceleration': total_l_heel_acceleration,
    'Total_R_Heel_Acceleration': total_r_heel_acceleration
})
heel_v_df = pd.DataFrame({
    'Time': time[1:],  # 总加速度的时间从第3个点开始
    'Total_L_Heel_V': total_l_heel_v,
    'Total_R_Heel_V': total_r_heel_v
})

l_toe_x = df['l_toe_X'].values * 1e-3
l_toe_y = df['l_toe_Y'].values * 1e-3
l_toe_z = df['l_toe_Z'].values * 1e-3

r_toe_x = df['r_toe_X'].values * 1e-3
r_toe_y = df['r_toe_Y'].values * 1e-3
r_toe_z = df['r_toe_Z'].values * 1e-3

# 获取时间列
time = df['Time'].values

# 计算左脚趾的位置变化（速度）
l_toe_velocity_x = np.diff(l_toe_x) / 0.008333
l_toe_velocity_y = np.diff(l_toe_y) / 0.008333
l_toe_velocity_z = np.diff(l_toe_z) / 0.008333

# 计算右脚趾的位置变化（速度）
r_toe_velocity_x = np.diff(r_toe_x) / 0.008333
r_toe_velocity_y = np.diff(r_toe_y) / 0.008333
r_toe_velocity_z = np.diff(r_toe_z) / 0.008333

# 计算左脚趾的加速度（加速度 = 速度变化 / 时间差）
l_toe_acceleration_x = np.diff(l_toe_velocity_x) / 0.008333
l_toe_acceleration_y = np.diff(l_toe_velocity_y) / 0.008333
l_toe_acceleration_z = np.diff(l_toe_velocity_z) / 0.008333

# 计算右脚趾的加速度（加速度 = 速度变化 / 时间差）
r_toe_acceleration_x = np.diff(r_toe_velocity_x) / 0.008333
r_toe_acceleration_y = np.diff(r_toe_velocity_y) / 0.008333
r_toe_acceleration_z = np.diff(r_toe_velocity_z) / 0.008333

# 计算左脚趾和右脚趾的总速度
# total_l_toe_v = np.sqrt(
#     l_toe_velocity_x**2 + l_toe_velocity_y**2 + l_toe_velocity_z**2
# )

# total_r_toe_v = np.sqrt(
#     r_toe_velocity_x**2 + r_toe_velocity_y**2 + r_toe_velocity_z**2
# )

# # 计算左脚趾和右脚趾的总加速度
# total_l_toe_acceleration = np.sqrt(
#     l_toe_acceleration_x**2 + l_toe_acceleration_y**2 + l_toe_acceleration_z**2
# )

# total_r_toe_acceleration = np.sqrt(
#     r_toe_acceleration_x**2 + r_toe_acceleration_y**2 + r_toe_acceleration_z**2
# )

total_l_toe_v = l_toe_velocity_x

total_r_toe_v = r_toe_velocity_x

total_l_toe_acceleration = l_toe_acceleration_x

total_r_toe_acceleration = r_toe_acceleration_x

# 创建一个新的DataFrame来存储加速度数据
toe_acceleration_df = pd.DataFrame({
    'Time': time[2:],  # 总加速度的时间从第3个点开始
    'Total_L_Toe_Acceleration': total_l_toe_acceleration,
    'Total_R_Toe_Acceleration': total_r_toe_acceleration
})

# 创建一个新的DataFrame来存储速度数据
toe_v_df = pd.DataFrame({
    'Time': time[1:],  # 总速度的时间从第2个点开始
    'Total_L_Toe_V': total_l_toe_v,
    'Total_R_Toe_V': total_r_toe_v
})

# 可视化总加速度数据（左右脚跟）
plt.figure(figsize=(16, 16))

plt.subplot(2, 2, 1)
plt.plot(heel_v_df['Time'], heel_v_df['Total_L_Heel_V'], color='purple', label='Left Heel Total Velocity')
plt.plot(heel_v_df['Time'], heel_v_df['Total_R_Heel_V'], color='blue', label='Right Heel Total Velocity')
plt.title('Left and Right Heel Total Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Total Acceleration (m/s)')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(toe_v_df['Time'], toe_v_df['Total_L_Toe_V'], color='purple', label='Left Toe Total Velocity')
plt.scatter(toe_v_df['Time'], toe_v_df['Total_R_Toe_V'], color='blue', label='Right Toe Total Velocity')
plt.title('Left and Right Toe Total Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Total Acceleration (m/s)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(heel_acceleration_df['Time'], heel_acceleration_df['Total_L_Heel_Acceleration'], color='purple', label='Left Heel Total Acceleration')
plt.plot(heel_acceleration_df['Time'], heel_acceleration_df['Total_R_Heel_Acceleration'], color='blue', label='Right Heel Total Acceleration')
plt.title('Left and Right Heel Total Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Total Acceleration (m/s²)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(toe_acceleration_df['Time'], toe_acceleration_df['Total_L_Toe_Acceleration'], color='purple', label='Left Toe Total Acceleration')
plt.plot(toe_acceleration_df['Time'], toe_acceleration_df['Total_R_Toe_Acceleration'], color='blue', label='Right Toe Total Acceleration')
plt.title('Left and Right Toe Total Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Total Acceleration (m/s²)')
plt.legend()

plt.tight_layout()
plt.show(block = 0)

# imu_acc_data = np.load(f'{address}saved_acc.npy')

# # 提取时间，左脚加速度，右脚加速度
# time_imu = imu_acc_data[:, 0]  # 第1列是时间
# time_imu = time_imu - imu_acc_data[0, 0]  # 将时间调整为相对时间，基于第一个时间点
# l_heel_acc = imu_acc_data[:, 1]  # 第2列是左脚加速度
# r_heel_acc = imu_acc_data[:, 2]  # 第3列是右脚加速度

# # 计算总加速度
# total_l_heel_acc = np.sqrt(l_heel_acc**2)  # 左脚总加速度
# total_r_heel_acc = np.sqrt(r_heel_acc**2)  # 右脚总加速度

# total_l_heel_acc = l_heel_acc
# total_r_heel_acc = r_heel_acc

# # 设置高斯滤波的标准差（sigma）
# sigma = 2.0

# # 对左脚和右脚的加速度进行高斯滤波
# filtered_l_heel_acc = gaussian_filter1d(total_l_heel_acc, sigma=sigma)
# filtered_r_heel_acc = gaussian_filter1d(total_r_heel_acc, sigma=sigma)

# # 初始化绘图
# plt.figure(figsize=(12, 8))

# # 绘制原始加速度
# plt.subplot(2, 1, 1)  # 2行1列，第1个子图
# plt.plot(time_imu, total_l_heel_acc, color='purple', label='Left Heel Acceleration (Raw)')
# plt.plot(time_imu, total_r_heel_acc, color='blue', label='Right Heel Acceleration (Raw)')
# plt.title('Left and Right Heel Acceleration (Raw)')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (m/s²)')
# plt.legend()

# # 绘制经过高斯滤波后的加速度
# plt.subplot(2, 1, 2)  # 2行1列，第2个子图
# plt.plot(time_imu, filtered_l_heel_acc, color='purple', label='Left Heel Acceleration (Filtered)')
# plt.plot(time_imu, filtered_r_heel_acc, color='blue', label='Right Heel Acceleration (Filtered)')
# plt.title('Left and Right Heel Acceleration (Filtered with Gaussian)')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (m/s²)')
# plt.legend()

# # 调整布局并显示图形
# plt.tight_layout()
# plt.show(block = 0)

# buf = total_r_heel_acceleration
# peaks = detect_peaks(-buf,show=True)
# cond = -buf[peaks]
# peaks = peaks[(cond > -1.91) & (cond < -1.82)] #czk_1
# # peaks = peaks[(cond > -1.87) & (cond < -1.73)] #czk_2
# # peaks = peaks[(cond > -2.59) & (cond < -2.46)] #czk_3
# # peaks = peaks[(cond > -1.85) & (cond < -1.62)] #czk_4
# # peaks = peaks[(cond > -2.73) & (cond < -2.55)] #czk_5
# # peaks = peaks[(cond > -2.7) & (cond < -2.61)] #czk_5
# print(peaks,time[peaks[0]])
# time_maker = time[peaks[0]]

# buf = filtered_r_heel_acc
# peaks = detect_peaks(-buf,show=True)
# cond = -buf[peaks]
# peaks = peaks[(cond > -1.9) & (cond < -1.82)] #czk_1
# # peaks = peaks[(cond > -1.87) & (cond < -1.73)] #czk_2
# # peaks = peaks[(cond > -2.59) & (cond < -2.40)] #czk_3
# # peaks = peaks[(cond > -1.85) & (cond < -1.45)] #czk_4
# # peaks = peaks[(cond > -2.95) & (cond < -2.63)] #czk_5
# # peaks = peaks[(cond > -2.93) & (cond < -2.70)] #czk_5
# print(peaks,time_imu[peaks[0]])
# time_imus = time_imu[peaks[0]]
# print('bias_time:',time_imus - time_maker)

# # buf = total_r_heel_acceleration
# # peaks = detect_peaks(buf,show=0)
# # cond = buf[peaks]
# # # peaks = peaks[(cond > 11.59) & (cond < 12.45)] #czk_3_l
# # # peaks = peaks[(cond > 10.64) & (cond < 10.99)] #xhy_1
# # # peaks = peaks[(cond > 4.94) & (cond < 5.05)] #xhy_2
# # # peaks = peaks[(cond > 5.00) & (cond < 5.32)] #xhy_3
# # # peaks = peaks[(cond > 10.75) & (cond < 11)] #xhy_4
# # # peaks = peaks[(cond > 8.75) & (cond < 8.93)] #xhy_5
# # peaks = peaks[(cond > 13.32) & (cond < 13.49)] #xhy_6
# # print(peaks,time[peaks[0]])
# # time_maker = time[peaks[0]]

# # buf = filtered_r_heel_acc
# # peaks = detect_peaks(buf,show=0)
# # cond = buf[peaks]
# # # peaks = peaks[(cond > 19) & (cond < 20.5)] #czk_3_l
# # # peaks = peaks[(cond > 9.45) & (cond < 11.43)] #xhy_1
# # # peaks = peaks[(cond > 4.32) & (cond < 4.60)] #xhy_2
# # # peaks = peaks[(cond > 4.67) & (cond < 4.90)] #xhy_3
# # # peaks = peaks[(cond > 9.94) & (cond < 10.5)] #xhy_4
# # # peaks = peaks[(cond > 8.66) & (cond < 8.71)] #xhy_5
# # peaks = peaks[(cond > 13.32) & (cond < 13.65)] #xhy_6
# # # print(peaks)
# # print(peaks,time_imu[peaks[0]])
# # time_imus = time_imu[peaks[0]]
# # print('bias_time:',time_imus - time_maker)

data = pd.read_csv(f'{address}Trimmed_exp_{sub}.forces', delim_whitespace=True, skiprows=4)

# 提取时间和Z轴受力数据
time = data['Sample'] / 1200  # 假设Sample是根据时间序列，从1到5350，且SampleRate为1200，单位为秒
force_Z1 = data['FZ1'].values 
force_Z2 = data['FZ2'].values

if sub == 'czk_5':
    force_Z1[(time>4.13)&(time<4.74)] = 0

if sub == 'xhy_1':
    force_Z1[(time>4.008)&(time<4.71)] = 0

if sub == 'xhy_3':
    force_Z1[(time>3.80)&(time<4.539)] = 0
    
if sub == 'xhy_6':
    force_Z1[(time>3.479)&(time<4.136)] = 0

force_Z1 = gaussian_filter1d(force_Z1, sigma=10)
force_Z2 = gaussian_filter1d(force_Z2, sigma=10)

# 绘制Z轴受力情况
plt.figure(figsize=(10, 6))
time = data['Sample'] / 1200
plt.plot(time, force_Z1, label='Force Plate 1 (FZ1)', color='blue')
plt.plot(time, force_Z2, label='Force Plate 2 (FZ2)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Z-Axis Force (N)')
plt.title('Z-Axis Force Measurement from Two Force Plates')
plt.legend()
plt.grid(True)
plt.show(block = 1)

# force_Z1[force_Z1>100] =0
# force_Z2[force_Z2>100] =0

buf_l = []
buf_r = []
tf_l = []
tf_r = []
hs_l = []
hs_r = []
win_size = 3
idx = 0

for i in range(len(time)-win_size):
    idx+=1
    buf_l = force_Z1[i:i+win_size]
    buf_r = force_Z2[i:i+win_size]
    if idx > 5:
        if buf_l[2] > 0 and buf_l[0] <= 0 and abs(buf_l[1]< 1e-3) and force_Z1[i+100]>2:
            hs_l.append(time[i])
            idx = 0
        if buf_r[2] > 0 and buf_r[0] <= 0 and abs(buf_r[1]< 1e-3) and force_Z2[i+100]>2:
            hs_r.append(time[i])
            idx = 0
        if buf_l[2] <= 0 and buf_l[0] > 0 and abs(buf_l[1]< 1e-3) and force_Z1[i-100]>2:
            tf_l.append(time[i])
            idx = 0
        if buf_r[2] <= 0 and buf_r[0] > 0 and abs(buf_r[1]< 1e-3) and force_Z2[i-100]>2:
            tf_r.append(time[i])
            idx = 0

time = df['Frame#'].values[1:]/120
# idx = np.argmin(time -tf_l[0])
# print(tf_l)
toe_v_l = total_l_toe_v[np.argmin(abs(time -tf_l[0]))]
toe_v_r = total_r_toe_v[np.argmin(abs(time -tf_r[0]))]
# print(np.argmin(time -tf_l[0]),np.argmin(time -tf_r[0]))
# print(tf_l)
tf_l.append(time[abs(total_l_toe_v-toe_v_l) < 0.1][0])
tf_r.append(time[abs(total_r_toe_v-toe_v_r) < 0.1][0])
tf_l[0],tf_l[1] = tf_l[1],tf_l[0]
tf_r[0],tf_r[1] = tf_r[1],tf_r[0]
tf_l = np.array(tf_l)
tf_r = np.array(tf_r)
hs_l = np.array(hs_l)
hs_r = np.array(hs_r)

print(f'right toe_off time: {tf_r}')
print(f'left toe_off time: {tf_l}')
print(f'right heel_strike time: {hs_r}')
print(f'left heel_strike time: {hs_l}\r\n')

import os
import copy

base_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(base_dir, f"result/gait_event_time_{sub}.yaml")

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

saved_time = {'l_tf':tf_l.tolist(),'r_tf':tf_r.tolist(),'l_hs':hs_l.tolist(),'r_hs':hs_r.tolist()}
tf_l_err = tf_l-np.array(data['IMU_recorded_data']['l_tf'])
tf_r_err = tf_r-np.array(data['IMU_recorded_data']['r_tf'])
hs_l_err = hs_l-np.array(data['IMU_recorded_data']['l_hs'])
hs_r_err = hs_r-np.array(data['IMU_recorded_data']['r_hs'])
error_time = {'l_tf':tf_l_err.tolist(),'r_tf':tf_r_err.tolist(),
              'l_hs':hs_l_err.tolist(),'r_hs':hs_r_err.tolist()}
avg_time_err = {'l_tf':np.mean(tf_l-np.array(data['IMU_recorded_data']['l_tf'])).tolist(),'r_tf':np.mean(tf_r-np.array(data['IMU_recorded_data']['r_tf'])).tolist(),
              'l_hs':np.mean(hs_l-np.array(data['IMU_recorded_data']['l_hs'])).tolist(),'r_hs':np.mean(hs_r-np.array(data['IMU_recorded_data']['r_hs'])).tolist()}
updated_data = {'IMU_recorded_data':data['IMU_recorded_data'],'Maker_recorded_data':saved_time,'Time_error':error_time,'Avg_time_error':avg_time_err}

tf_l -= np.array(data['IMU_recorded_data']['l_tf'])
tf_r -= np.array(data['IMU_recorded_data']['r_tf'])
hs_l -= np.array(data['IMU_recorded_data']['l_hs'])
hs_r -= np.array(data['IMU_recorded_data']['r_hs'])

print(f'right toe_off time error: {tf_r}')
print(f'left toe_off time error: {tf_l}')
print(f'right heel_strike time error: {hs_r}')
print(f'left heel_strike time error: {hs_l}\r\n')

with open(yaml_path, "w") as f:
    yaml.dump(updated_data, f, indent=2, sort_keys=False)


avg_time_err = []
sub = 'xhy'
for i in range(6):

    yaml_path = os.path.join(base_dir, f"result/gait_event_time_{sub}_{i+1}.yaml")
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    err = [data['Avg_time_error']['l_tf'],data['Avg_time_error']['r_tf'],data['Avg_time_error']['l_hs'],data['Avg_time_error']['r_hs']]
    avg_time_err.append(err)
avg_time_err = np.mean(np.array(avg_time_err),axis = 0)
print(avg_time_err)
