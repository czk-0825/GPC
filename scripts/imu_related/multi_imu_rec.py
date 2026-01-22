
import rospy
import numpy as np
import time 
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
import message_filters
import threading
from scipy.ndimage import gaussian_filter1d
import os
from detecta import detect_peaks
import copy
from samcon_perception.msg import Gait_Info
from std_msgs.msg import Header
import warnings
import yaml
warnings.filterwarnings("ignore", category=RuntimeWarning)
YELLOW = "\033[93m"
RESET = "\033[0m"

class Joint_state():
    def __init__(self):

        self.timestamp = None

        self.r_thigh_angle = 0
        self.l_thigh_angle = 0
        self.r_knee_angle = 0
        self.l_knee_angle = 0
        self.r_ankle_angle = 0
        self.l_ankle_angle = 0

        self.r_thigh_av = None
        self.l_thigh_av = None
        self.r_knee_av = None
        self.l_knee_av = None
        self.r_ankle_av = None
        self.l_ankle_av = None

        self.r_thigh_acc_z = None
        self.l_thigh_acc_z = None
        self.r_knee_acc_z = None
        self.l_knee_acc_z = None
        self.r_ankle_acc_z = None
        self.l_ankle_acc_z = None

        self.l_ankle_acc = None
        self.r_ankle_acc = None

class IMU_processor():
    def __init__(self):

        # 滑窗参数
        self.win_size = 50
        self.sig = 5

        self.l_slide_idx = 0
        self.r_slide_idx = 0 

        self.l_toe_off_angle = 0
        self.r_toe_off_angle = 0

        self.l_phase = 'stance'
        self.r_phase = 'stance'

        self.l_late_swing_idx = 0
        self.r_late_swing_idx = 0

        self.l_peaks = None
        self.r_peaks = None
        self.first_peak = None
        self.start_time = None

        self.l_toe_flag = 0
        self.r_toe_flag = 0
        self.l_hs_flag = 0
        self.r_hs_flag = 0

        self.is_gravity_init = False
        self.init_gra_num = 0
        self.gravity_lt = 0
        self.gravity_ls = 0
        self.gravity_lf = 0
        self.gravity_rt = 0
        self.gravity_rs = 0
        self.gravity_rf = 0
        
        self.init_joint_state = Joint_state()
        self.joint_state_buf = []

        self.rta_buf = []
        self.lta_buf = []
        self.rka_buf = []
        self.lka_buf = []
        self.raa_buf = []
        self.laa_buf = []

        self.lt_av_buf = []
        self.rt_av_buf = []
        self.rk_av_buf = []
        self.lk_av_buf = []
        self.ra_av_buf = []
        self.la_av_buf = []
        self.la_acc_buf = []

        self.l_foot_imu_av_buf = []
        self.r_foot_imu_av_buf = []

        self.condition = threading.Condition()
        self.gpp_condition = threading.Condition()

        self.save_data = {'l_tf':[],'r_tf':[],'l_hs':[],'r_hs':[]}
        self.save_imu_data = []

        self._start_imu()

        # self._init_imu()

    def _start_imu(self):
        self.event_pub = rospy.Publisher(f"imu/gait_event",Gait_Info, queue_size = 5)

        self.imu_lt = message_filters.Subscriber("/imu/left_thigh", Imu)
        self.imu_ls = message_filters.Subscriber("/imu/left_shank", Imu)
        self.imu_lf = message_filters.Subscriber("/imu/left_foot", Imu)
        self.imu_rt = message_filters.Subscriber("/imu/right_thigh", Imu)
        self.imu_rs = message_filters.Subscriber("/imu/right_shank", Imu)
        self.imu_rf = message_filters.Subscriber("/imu/right_foot", Imu)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.imu_lt, self.imu_ls, self.imu_lf, self.imu_rt, self.imu_rs, self.imu_rf], 5, 0.005 )
        self.ts.registerCallback(self.imu_callback)


    def quat2euler(self,imu_msg):
        quat = [imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w]
        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        return euler,quat
    
    
    def init_gravity_joint(self,lt, ls, lf, rt, rs, rf, threshold = 30):
        self.gravity_lt += np.linalg.norm(np.array([lt.linear_acceleration.x,lt.linear_acceleration.y,lt.linear_acceleration.z]))
        self.gravity_ls += np.linalg.norm(np.array([ls.linear_acceleration.x,ls.linear_acceleration.y,ls.linear_acceleration.z]))
        self.gravity_lf += np.linalg.norm(np.array([lf.linear_acceleration.x,lf.linear_acceleration.y,lf.linear_acceleration.z]))
        self.gravity_rt += np.linalg.norm(np.array([rt.linear_acceleration.x,rt.linear_acceleration.y,rt.linear_acceleration.z]))
        self.gravity_rs += np.linalg.norm(np.array([rs.linear_acceleration.x,rs.linear_acceleration.y,rs.linear_acceleration.z]))
        self.gravity_rf += np.linalg.norm(np.array([rf.linear_acceleration.x,rf.linear_acceleration.y,rf.linear_acceleration.z]))
        self.init_gra_num+=1

        lt_euler,lt_quat = self.quat2euler(lt)
        ls_euler,ls_quat = self.quat2euler(ls)
        lf_euler,lf_quat = self.quat2euler(lf)
        rt_euler,rt_quat = self.quat2euler(rt)
        rs_euler,rs_quat = self.quat2euler(rs)
        rf_euler,rf_quat = self.quat2euler(rf)

        self.init_joint_state.l_thigh_angle += lt_euler[0]
        self.init_joint_state.r_thigh_angle += rt_euler[0]
        self.init_joint_state.l_knee_angle += lt_euler[0] - ls_euler[0]
        self.init_joint_state.r_knee_angle += rt_euler[0] - rs_euler[0]
        self.init_joint_state.l_ankle_angle += ls_euler[0] - lf_euler[0]
        self.init_joint_state.r_ankle_angle += rs_euler[0] - rf_euler[0]

        if self.init_gra_num == threshold:
            self.gravity_lt /= threshold
            self.gravity_ls /= threshold
            self.gravity_lf /= threshold
            self.gravity_rt /= threshold
            self.gravity_rs /= threshold
            self.gravity_rf /= threshold

            self.init_joint_state.l_thigh_angle /= threshold
            # print(self.init_joint_state.l_thigh_angle)
            self.init_joint_state.r_thigh_angle /= threshold
            self.init_joint_state.l_knee_angle /= threshold
            # print(self.init_joint_state.l_knee_angle)
            self.init_joint_state.r_knee_angle /= threshold
            self.init_joint_state.l_ankle_angle /= threshold
            self.init_joint_state.r_ankle_angle /= threshold

            self.is_gravity_init = True 

    def imu_callback(self,lt, ls, lf, rt, rs, rf):

        t = time.time()

        if not self.is_gravity_init:
            self.init_gravity_joint(lt, ls, lf, rt, rs, rf)
            return
        
        if self.start_time is None:
            self.start_time = lt.header.stamp.to_sec()

        lt_euler,lt_quat = self.quat2euler(lt)
        ls_euler,ls_quat = self.quat2euler(ls)
        lf_euler,lf_quat = self.quat2euler(lf)
        rt_euler,rt_quat = self.quat2euler(rt)
        rs_euler,rs_quat = self.quat2euler(rs)
        rf_euler,rf_quat = self.quat2euler(rf)

        js = Joint_state()

        js.timestamp = lt.header.stamp.to_sec()
        # save_acc[0] = js.timestamp

        js.l_thigh_angle = lt_euler[0] - self.init_joint_state.l_thigh_angle
        js.r_thigh_angle = rt_euler[0] - self.init_joint_state.r_thigh_angle
        js.l_knee_angle = lt_euler[0] - ls_euler[0] - self.init_joint_state.l_knee_angle
        if js.l_knee_angle < -180:
            js.l_knee_angle += 360
        js.r_knee_angle = rt_euler[0] - rs_euler[0] - self.init_joint_state.r_knee_angle
        if js.r_knee_angle < -180:
            js.r_knee_angle += 360
        js.l_ankle_angle = ls_euler[0] - lf_euler[0] - self.init_joint_state.l_ankle_angle
        if js.l_ankle_angle < -180:
            js.l_ankle_angle += 360
        js.r_ankle_angle = rs_euler[0] - rf_euler[0] - self.init_joint_state.r_ankle_angle
        if js.r_ankle_angle < -180:
            js.r_ankle_angle += 360

        js.r_thigh_av = rt.angular_velocity.x
        js.l_thigh_av = lt.angular_velocity.x
        js.r_knee_av = rt.angular_velocity.x - rs.angular_velocity.x
        js.l_knee_av = lt.angular_velocity.x - ls.angular_velocity.x
        js.r_ankle_av = rs.angular_velocity.x - rf.angular_velocity.x
        js.l_ankle_av = ls.angular_velocity.x - lf.angular_velocity.x
        # self.plus_value = - lf.angular_velocity.x
        # print(lf.angular_velocity.x)

        js.r_thigh_acc_z = rt.linear_acceleration.z - (R.from_quat(rt_quat).as_matrix().T @ np.array([0,0,self.gravity_rt]))[2]
        js.l_thigh_acc_z = lt.linear_acceleration.z - (R.from_quat(lt_quat).as_matrix().T @ np.array([0,0,self.gravity_lt]))[2]
        js.r_knee_acc_z = rs.linear_acceleration.z - (R.from_quat(rs_quat).as_matrix().T @ np.array([0,0,self.gravity_rs]))[2]
        js.l_knee_acc_z = ls.linear_acceleration.z - (R.from_quat(ls_quat).as_matrix().T @ np.array([0,0,self.gravity_ls]))[2]
        js.r_ankle_acc_z = rf.linear_acceleration.z - (R.from_quat(rf_quat).as_matrix().T @ np.array([0,0,self.gravity_rf]))[2]
        js.l_ankle_acc_z = lf.linear_acceleration.z - (R.from_quat(lf_quat).as_matrix().T @ np.array([0,0,self.gravity_lf]))[2]

        js.l_ankle_acc = np.linalg.norm(np.array([lf.linear_acceleration.x,lf.linear_acceleration.y,lf.linear_acceleration.z])
                                         - (R.from_quat(lf_quat).as_matrix().T @ np.array([0,0,self.gravity_lf])))
        js.r_ankle_acc = np.linalg.norm(np.array([rf.linear_acceleration.x,rf.linear_acceleration.y,rf.linear_acceleration.z])
                                         - (R.from_quat(rf_quat).as_matrix().T @ np.array([0,0,self.gravity_rf])))
        # save_acc[1] = js.l_ankle_acc
        # save_acc[2] = js.r_ankle_acc
        # self.save_data.append(save_acc)

        self.joint_state_buf.append(js)
        self.rta_buf.append(js.r_thigh_angle)
        self.lta_buf.append(js.l_thigh_angle)
        self.rka_buf.append(js.r_knee_angle)
        self.lka_buf.append(js.l_knee_angle)
        self.raa_buf.append(js.r_ankle_angle)
        self.laa_buf.append(js.l_ankle_angle)

        self.rt_av_buf.append(js.r_thigh_av)
        self.lt_av_buf.append(js.l_thigh_av)
        self.rk_av_buf.append(js.r_knee_av)
        self.lk_av_buf.append(js.l_knee_av)
        self.ra_av_buf.append(js.r_ankle_av)
        self.la_av_buf.append(js.l_ankle_av)
        self.la_acc_buf.append(js.l_ankle_acc_z)
        
        self.l_foot_imu_av_buf.append(-lf.angular_velocity.x)
        self.r_foot_imu_av_buf.append(-rf.angular_velocity.x)

        if len(self.joint_state_buf)> self.win_size:

            self.joint_state_buf = self.joint_state_buf[1:]
            self.rta_buf = self.rta_buf[1:]
            self.lta_buf = self.lta_buf[1:]
            self.rka_buf = self.rka_buf[1:]
            self.lka_buf = self.lka_buf[1:]
            self.raa_buf = self.raa_buf[1:]
            self.laa_buf = self.laa_buf[1:]

            self.rt_av_buf = self.rt_av_buf[1:]
            self.lt_av_buf = self.lt_av_buf[1:]
            self.rk_av_buf = self.rk_av_buf[1:]
            self.lk_av_buf = self.lk_av_buf[1:]
            self.ra_av_buf = self.ra_av_buf[1:]
            self.la_av_buf = self.la_av_buf[1:]

            self.la_acc_buf = self.la_acc_buf[1:]

            self.l_foot_imu_av_buf = self.l_foot_imu_av_buf[1:]
            self.r_foot_imu_av_buf = self.r_foot_imu_av_buf[1:]

        # lk_av_buf = gaussian_filter1d(self.lk_av_buf, sigma=self.sig)
        # rk_av_buf = gaussian_filter1d(self.rk_av_buf, sigma=self.sig)
        # ra_av_buf = gaussian_filter1d(self.ra_av_buf, sigma=self.sig)
        # la_av_buf = gaussian_filter1d(self.la_av_buf, sigma=self.sig)
        # la_acc_buf = gaussian_filter1d(self.la_acc_buf, sigma=self.sig)
        l_foot_imu_av_buf = gaussian_filter1d(self.l_foot_imu_av_buf, sigma=self.sig)
        r_foot_imu_av_buf = gaussian_filter1d(self.r_foot_imu_av_buf, sigma=self.sig)
        # save_acc[1] = l_foot_imu_av_buf[-1]
        # save_acc[2] = r_foot_imu_av_buf[-1]
        # self.save_data.append(save_acc)
        # self.plus_buf = gaussian_filter1d(self.la_av_buf, sigma=self.sig) + gaussian_filter1d(self.lk_av_buf, sigma=self.sig) - gaussian_filter1d(self.lt_av_buf, sigma=self.sig)

        self.gait_phase_det(js,l_foot_imu_av_buf,side = 'left')
        self.gait_phase_det(js,r_foot_imu_av_buf,side = 'right')
        # self.gait_phase_det(js,la_acc_buf)

        with self.condition:
            self.condition.notify()

        # print(time.time()-t)

    def gait_phase_det(self,current_js,buf,side = 'left'):
        # l_buf = copy.copy(self.la_av_buf)
        buf = np.array(buf)
        
        if side == 'left':
            self.l_slide_idx += 1
            phase = self.l_phase
        else:
            self.r_slide_idx += 1
            phase = self.r_phase

        if self.first_peak is not None:
            
            if phase == 'stance':
                peaks = detect_peaks(buf,mph = 75,mpd=40)
            else:
                peaks = detect_peaks(buf,mph = 15)

            if len(peaks)>1:
                value = buf[peaks]
                peaks = [peaks[np.argmax(value)]]

            if len(peaks) == 1 and (abs(np.argmax(buf)-peaks[0]) > 5 or buf[peaks[0]] < 15):
                peaks = []

            if side == 'left':
                self.l_peaks = peaks
            else:
                self.r_peaks = peaks
        else:
            peaks = detect_peaks(buf,mph = 10)

        # if side == 'left' and phase != 'stance' and len(self.r_peaks) !=0:
        #             if self.r_peaks[0] < 0.9*self.win_size:
        #                 print(f'left heel strike== time:{time.time()}\r\n')
        #                 self.pub_event(side = 'left',gait_event = 'heel_strike')
        #                 self.l_phase = "stance" 
        #                 self.l_slide_idx = 0
        #                 self.r_peaks = []
        # elif side == 'right' and phase != 'stance'and len(self.l_peaks) !=0:
        #             if self.l_peaks[0] < 0.9*self.win_size:
        #                 print(f'{YELLOW}right heel strike== time:{time.time()}{RESET}\r\n')
        #                 self.pub_event(side = 'right',gait_event = 'heel_strike')
        #                 self.r_phase = "stance" 
        #                 self.r_slide_idx = 0
        #                 self.l_peaks = []

        if (side == 'left' and self.l_slide_idx >= self.win_size*1.82) or (side == 'right' and self.r_slide_idx >= self.win_size*1.82):

            if len(peaks) != 0 or phase == 'late_swing':

                if phase != 'stance' and len(peaks) != 0:

                    # if side == 'left' and current_js.l_thigh_angle > self.l_toe_off_angle + 5:
                    if side == 'left':
                        self.save_data['l_hs'].append(current_js.timestamp - self.start_time)
                        print(f'left heel strike time:{current_js.timestamp - self.start_time}\r\n')
                        self.pub_event(side = 'left',gait_event = 'heel_strike')
                        self.l_phase = 'stance'
                        self.l_slide_idx = 0
                    # if side == 'right' and current_js.r_thigh_angle > self.r_toe_off_angle +5:
                    if side == 'right':
                        self.save_data['r_hs'].append(current_js.timestamp - self.start_time)
                        print(f'{YELLOW}right heel strike time:{current_js.timestamp - self.start_time}{RESET}\r\n')
                        self.pub_event(side = 'right',gait_event = 'heel_strike')
                        self.r_phase = 'stance'
                        self.r_slide_idx = 0

                if phase == 'stance':
                    if side == 'left' and current_js.l_knee_angle > 0 and current_js.l_thigh_av > -100:
                        if self.first_peak is None or peaks[0] < self.win_size*1:
                            self.save_data['l_tf'].append(current_js.timestamp - self.start_time)
                            print(f'left toe_off time:{current_js.timestamp - self.start_time}')
                            self.l_toe_off_angle = current_js.l_thigh_angle
                            if self.first_peak is None:
                                self.l_toe_off_angle = -100
                            self.l_phase = 'swing'
                            self.l_slide_idx = 0
                            self.pub_event(side = 'left',gait_event = 'toe_off')
                        
                    if side == 'right' and current_js.r_knee_angle > 0 and current_js.r_thigh_av > -100:
                        if self.first_peak is None or peaks[0] < self.win_size*1:
                            self.save_data['r_tf'].append(current_js.timestamp - self.start_time)
                            print(f'{YELLOW}right toe_off time:{current_js.timestamp - self.start_time}{RESET}')
                            self.r_toe_off_angle = current_js.r_thigh_angle
                            if self.first_peak is None:
                                self.r_toe_off_angle = -100
                            self.r_phase = 'swing'
                            self.r_slide_idx = 0
                            self.pub_event(side = 'right',gait_event = 'toe_off')

                # if np.mean(abs(buf[-int(self.win_size/2):])) < 3:
                if side == 'left' and phase != 'stance' and len(self.r_peaks) !=0:
                    if self.r_peaks[0] < 0.9*self.win_size:
                        print(f'left heel strike== time:{current_js.timestamp - self.start_time}\r\n')
                        self.pub_event(side = 'left',gait_event = 'heel_strike')
                        self.l_phase = "stance" 
                        self.l_slide_idx = 0
                        self.r_peaks = []
                elif side == 'right' and phase != 'stance'and len(self.l_peaks) !=0:
                    if self.l_peaks[0] < 0.9*self.win_size:
                        print(f'{YELLOW}right heel strike== time:{current_js.timestamp - self.start_time}{RESET}\r\n')
                        self.pub_event(side = 'right',gait_event = 'heel_strike')
                        self.r_phase = "stance" 
                        self.r_slide_idx = 0
                        self.l_peaks = []
            
            if phase == 'swing':
                valley = detect_peaks(-buf,mph = 50,mpd=40)
                # if len(valley) != 0:
                #     print(valley)
                if len(valley) != 0 and side == 'left':
                    self.r_peaks = []
                    self.l_late_swing_idx += 1
                    # if self.l_late_swing_idx > int(self.win_size/2):
                    if True:
                        self.l_phase = 'late_swing'
                        self.l_late_swing_idx = 0
                elif len(valley) != 0 and side == 'right':
                    self.l_peaks = []
                    self.r_late_swing_idx += 1
                    # if self.r_late_swing_idx > int(self.win_size/2):
                    if True:
                        self.r_phase = 'late_swing'
                        self.r_late_swing_idx = 0
                else:
                    if np.mean(abs(buf)) < 1:
                        if side == 'left':
                            self.l_slide_idx = 0
                            self.l_phase = "stance" 
                            self.pub_event(side = 'left',gait_event = 'heel_strike')
                            print(f'left set to stance time:{current_js.timestamp - self.start_time}\r\n')
                        elif side == 'right':
                            self.r_slide_idx = 0
                            self.r_phase = "stance" 
                            self.pub_event(side = 'right',gait_event = 'heel_strike')
                            print(f'{YELLOW}right set to stance time:{current_js.timestamp - self.start_time}{RESET}\r\n')

    def pub_event(self,side,gait_event):

        if self.first_peak is None:
            self.first_peak = 1
        
        if side == 'left' and gait_event == 'toe_off':
            self.l_toe_flag = 1
        if side == 'left' and gait_event == 'heel_strike':
            self.l_hs_flag = 1
        if side == 'right' and gait_event == 'toe_off':
            self.r_toe_flag = 1
        if side == 'right' and gait_event == 'heel_strike':
            self.r_hs_flag = 1

        event_msg = Gait_Info()

        header = Header(stamp=rospy.Time.now())
        header.frame_id = "body"

        event_msg.header = header
        event_msg.event = gait_event
        event_msg.side = side

        self.event_pub.publish(event_msg)

    def release_source(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(base_dir)
        global save_imu_data
        # np.save(base_dir+'/CZK/saved_imu_data.npy',self.save_imu_data)
        sub = 'czk_5'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, f"result/gait_event_time_{sub}.yaml")
        report = {'IMU_recorded_data':self.save_data}
        # with open(output_path, "w") as f:
        #     yaml.dump(report, f, indent=2, sort_keys=False)
        with self.condition:
            self.condition.notify()


if __name__ == "__main__":

    rospy.init_node("imu_processor")
    queue_size = 5
    processor = IMU_processor()
    rospy.on_shutdown(processor.release_source)

    rospy.spin()
    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(1200, 1200) 
    WINDOW_SIZE = 500
        
    p1 = win.addPlot(title="Left Thigh Angle")
    p2 = win.addPlot(title="Left Knee and Ankle Angles")
    # p5 = win.addPlot(title="Left Ankle Acc")
    win.nextRow()
    p3 = win.addPlot(title="Left Thigh Av")
    p4 = win.addPlot(title="Left Ankle and Knee Av")
    win.nextRow()
    p6 = win.addPlot(title="Full Av Plot", colspan=2)
    win.nextRow()
    p7 = win.addPlot(title="Ankle Acc Plot")

    p1.setYRange(-90, 90)
    p2.setYRange(-90, 90)
    p3.setYRange(-400, 400)
    p4.setYRange(-400, 400)
    p6.setYRange(-400, 400)
    p7.setYRange(-20, 20)

    p1.setLabel('left', 'Angle (°)')
    p1.setLabel('bottom', 'Time (s)')
    data1 = np.zeros(WINDOW_SIZE)  
    smoothed_1 = np.zeros(WINDOW_SIZE)  
    curve1 = p1.plot(data1, pen=pg.mkPen(color=(255, 0, 0), width=2)) 

    p2.setLabel('left', 'Angle (°)')
    p2.setLabel('bottom', 'Time (s)')
    data2 = np.zeros(WINDOW_SIZE)  
    data3 = np.zeros(WINDOW_SIZE)  
    smoothed_2 = np.zeros(WINDOW_SIZE) 
    smoothed_3 = np.zeros(WINDOW_SIZE) 
    curve2 = p2.plot(data2, pen=pg.mkPen(color=(0, 255, 0), width=2))  
    curve3 = p2.plot(data3, pen=pg.mkPen(color=(0, 0, 255), width=2))  

    p3.setLabel('left', 'Av (°/s)')
    p3.setLabel('bottom', 'Time (s)')
    data4 = np.zeros(WINDOW_SIZE)  
    smoothed_4 = np.zeros(WINDOW_SIZE) 
    curve4 = p3.plot(data4, pen=pg.mkPen(color=(255, 0, 0), width=2)) 
        
    p4.setLabel('left', 'Av (°/s)')
    p4.setLabel('bottom', 'Time (s)')
    data5 = np.zeros(WINDOW_SIZE)  
    smoothed_5 = np.zeros(WINDOW_SIZE) 
    curve5 = p4.plot(data5, pen=pg.mkPen(color=(0, 255, 0), width=2))  
    data6 = np.zeros(WINDOW_SIZE)  
    smoothed_6 = np.zeros(WINDOW_SIZE) 
    curve6 = p4.plot(data6, pen=pg.mkPen(color=(0, 0, 255), width=2))

    p6.setLabel('left', 'Av (°/s)')
    p6.setLabel('bottom', 'Time (s)')
    data7 = np.zeros(WINDOW_SIZE)
    smoothed_7 = np.zeros(WINDOW_SIZE) 
    data8 = np.zeros(WINDOW_SIZE)
    smoothed_8 = np.zeros(WINDOW_SIZE) 
    l_toe_off_data = np.full(WINDOW_SIZE,np.nan)
    r_toe_off_data = np.full(WINDOW_SIZE,np.nan)
    l_hs_data = np.full(WINDOW_SIZE,np.nan)
    r_hs_data = np.full(WINDOW_SIZE,np.nan)
    curve_l_toe = p6.plot(l_toe_off_data, pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=7)
    curve_r_toe = p6.plot(r_toe_off_data, pen=None, symbol='o', symbolBrush=(0, 255, 0), symbolSize=7)
    curve_l_hs = p6.plot(l_hs_data, pen=None, symbol='s', symbolBrush=(255, 0, 0), symbolSize=7)
    curve_r_hs = p6.plot(r_hs_data, pen=None, symbol='s', symbolBrush=(0, 255, 0), symbolSize=7)
    curve7 = p6.plot(data7, pen=pg.mkPen(color=(0, 255, 0), width=3))  
    curve8 = p6.plot(data8, pen=pg.mkPen(color=(0, 0, 255), width=3))

    p7.setLabel('left', 'Acc (m/s^2)')
    p7.setLabel('bottom', 'Time (s)')
    data_l_acc = np.zeros(WINDOW_SIZE) 
    data_r_acc = np.zeros(WINDOW_SIZE) 
    curve_l_acc = p7.plot(data_l_acc, pen=pg.mkPen(color=(0, 255, 0), width=2))  
    curve_r_acc = p7.plot(data_r_acc, pen=pg.mkPen(color=(0, 0, 255), width=2))
    smoothed_9 = np.zeros(WINDOW_SIZE)
    smoothed_10 = np.zeros(WINDOW_SIZE)

    save_left = []
    save_right = []
    l_to = []
    l_hs = []
    r_to = []
    r_hs = []

    while not rospy.is_shutdown():

        t = time.time()

        with processor.condition:
            processor.condition.wait()

        data1[:-1] = data1[1:]
        data2[:-1] = data2[1:]
        data3[:-1] = data3[1:]
        data4[:-1] = data4[1:]
        data5[:-1] = data5[1:]
        data6[:-1] = data6[1:]
        data7[:-1] = data7[1:]
        data8[:-1] = data8[1:]
        l_toe_off_data[:-1] = l_toe_off_data[1:]
        r_toe_off_data[:-1] = r_toe_off_data[1:]
        l_hs_data[:-1] = l_hs_data[1:]
        r_hs_data[:-1] = r_hs_data[1:]
        data_l_acc[:-1] = data_l_acc[1:]
        data_r_acc[:-1] = data_r_acc[1:]

        data1[-1] = processor.joint_state_buf[-1].r_thigh_angle
        data2[-1] = processor.joint_state_buf[-1].r_knee_angle
        data3[-1] = processor.joint_state_buf[-1].r_ankle_angle

        data4[-1] = processor.joint_state_buf[-1].r_thigh_av
        data5[-1] = processor.joint_state_buf[-1].r_knee_av
        data6[-1] = processor.joint_state_buf[-1].r_ankle_av

        data7[-1] = processor.joint_state_buf[-1].l_ankle_av + processor.joint_state_buf[-1].l_knee_av - processor.joint_state_buf[-1].l_thigh_av
        data8[-1] = processor.joint_state_buf[-1].r_ankle_av + processor.joint_state_buf[-1].r_knee_av - processor.joint_state_buf[-1].r_thigh_av
        # processor.save_acc[1] = processor.l_foot_imu_av_buf[-1]
        # processor.save_acc[2] = processor._foot_imu_av_buf[-1]
        # processor.save_data.append(save_acc)

        l_toe_off_data[-1] = np.nan
        r_toe_off_data[-1] = np.nan
        l_hs_data[-1] = np.nan
        r_hs_data[-1] = np.nan

        data_l_acc[-1] = processor.joint_state_buf[-1].l_ankle_acc
        data_r_acc[-1] = processor.joint_state_buf[-1].r_ankle_acc

        sig = 5
        win_size = 35
        s_1 = gaussian_filter1d(data1[-win_size:], sigma=sig)[-1]
        s_2 = gaussian_filter1d(data2[-win_size:], sigma=sig)[-1]
        s_3 = gaussian_filter1d(data3[-win_size:], sigma=sig)[-1]
        s_4 = gaussian_filter1d(data4[-win_size:], sigma=sig)[-1] 
        s_5 = gaussian_filter1d(data5[-win_size:], sigma=sig)[-1] 
        s_6 = gaussian_filter1d(data6[-win_size:], sigma=sig)[-1] 
        s_7 = gaussian_filter1d(data7[-win_size:], sigma=sig)[-1] 
        s_8 = gaussian_filter1d(data8[-win_size:], sigma=sig)[-1] 
        s_9 = gaussian_filter1d(data_l_acc[-win_size:], sigma=1)[-1] 
        s_10 = gaussian_filter1d(data_r_acc[-win_size:], sigma=1)[-1] 

        smoothed_1[:-1] = smoothed_1[1:]
        smoothed_2[:-1] = smoothed_2[1:]
        smoothed_3[:-1] = smoothed_3[1:]
        smoothed_4[:-1] = smoothed_4[1:]
        smoothed_5[:-1] = smoothed_5[1:]
        smoothed_6[:-1] = smoothed_6[1:]
        smoothed_7[:-1] = smoothed_7[1:]
        smoothed_8[:-1] = smoothed_8[1:]
        smoothed_9[:-1] = smoothed_9[1:]
        smoothed_10[:-1] = smoothed_10[1:]
        # processor.save_data.append(smoothed_6)

        # valleys = detect_peaks(-smoothed_6[-50:],mph = 50)

        # print(valleys)

        smoothed_1[-1] = s_1
        smoothed_2[-1] = s_2
        smoothed_3[-1] = s_3
        smoothed_4[-1] = s_4
        smoothed_5[-1] = s_5
        smoothed_6[-1] = s_6
        smoothed_7[-1] = s_7
        smoothed_8[-1] = s_8
        smoothed_9[-1] = s_9
        smoothed_10[-1] = s_10

        # processor.save_imu_data.append(s_8)
        save_left.append(s_7)
        save_right.append(s_8)

        # curve1.setData(data1)
        # curve2.setData(data2)
        # curve3.setData(data3)
        # curve4.setData(data4)
        # plus = processor.plus_buf
        # curve5.setData(data6)
        # curve6.setData(data6)
        # curve7.setData(data7)

        if processor.l_toe_flag:
            l_toe_off_data[-1] = smoothed_7[-1]
            processor.l_toe_flag = 0
            l_to.append(smoothed_7[-1])
        else:
            l_to.append(1000)
        if processor.l_hs_flag:
            l_hs_data[-1] = smoothed_7[-1]
            processor.l_hs_flag = 0
            l_hs.append(smoothed_7[-1])
        else:
            l_hs.append(1000)
        if processor.r_toe_flag:
            r_toe_off_data[-1] = smoothed_8[-1]
            processor.r_toe_flag = 0
            r_to.append(smoothed_8[-1])
        else:
            r_to.append(1000)
        if processor.r_hs_flag:
            r_hs_data[-1] = smoothed_8[-1]
            processor.r_hs_flag = 0
            r_hs.append(smoothed_8[-1])
        else:
            r_hs.append(1000)

        curve1.setData(smoothed_1)
        curve2.setData(smoothed_2)
        curve3.setData(smoothed_3)
        curve4.setData(smoothed_4)
        curve5.setData(smoothed_5)
        curve6.setData(smoothed_6)
        curve7.setData(smoothed_7)
        curve8.setData(smoothed_8)
        curve_l_toe.setData(l_toe_off_data)
        curve_r_toe.setData(r_toe_off_data)
        curve_l_hs.setData(l_hs_data)
        curve_r_hs.setData(r_hs_data)
        curve_l_acc.setData(smoothed_9)
        curve_r_acc.setData(smoothed_10)

        QtWidgets.QApplication.processEvents()

        # print(time.time()-t)

    # np.save('left_heel.npy',save_left)
    # np.save('right_heel.npy',save_right)
    # np.save('lto.npy',l_to)
    # np.save('rto.npy',r_to)
    # np.save('lhs.npy',l_hs)
    # np.save('rhs.npy',r_hs)



