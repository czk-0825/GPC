# live_scene_and_gaze.py : A demo for video streaming and synchronized gaze
#
# Copyright (C) 2021  Davide De Tommaso
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
import time
from tobiiglassesctrl import TobiiGlassesController
import rospy
from std_msgs.msg import Header
from samcon_perception.msg import Glass_Gaze
from scipy.ndimage import gaussian_filter1d


rospy.init_node("gaze_publisher",anonymous=True)
gaze_pub = rospy.Publisher("/glass/gaze",Glass_Gaze,queue_size=10)
save_path = "/home/czk/桌面/pcd_data/"
ipv4_address = "192.168.71.50"

tobiiglasses = TobiiGlassesController(ipv4_address)
tobiiglasses.start_streaming()
rate = rospy.Rate(50)

# if tobiiglasses.is_recording():
# 	rec_id = tobiiglasses.get_current_recording_id()
# 	tobiiglasses.stop_recording(rec_id)

project_name = '1'
project_id = tobiiglasses.create_project(project_name)

participant_name = 'chenzhaokai'
participant_id = tobiiglasses.create_participant(project_id, participant_name)

calibration_id = tobiiglasses.create_calibration(project_id, participant_id)
input("Put the calibration marker in front of the user, then press enter to calibrate")
tobiiglasses.start_calibration(calibration_id)

res = tobiiglasses.wait_until_calibration_is_done(calibration_id)

if res is False:
	print("Calibration failed!")
	exit(1)

while not rospy.is_shutdown():

    try:
        t0 = time.time()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = "camera_color_optical_frame"
        data_gp = tobiiglasses.get_data()['gp']
        gaze_msg = Glass_Gaze()
        gaze_msg.header = header
        gaze_msg.width = 1080
        gaze_msg.height = 1920

        if data_gp['ts'] > 0:

            gaze_msg.u_2d = data_gp['gp'][1] * 1080
            gaze_msg.v_2d = data_gp['gp'][0] * 1920
            print('gaze point:',(gaze_msg.v_2d,gaze_msg.u_2d))
            gaze_pub.publish(gaze_msg)

        else:
            gaze_msg = Glass_Gaze()
            gaze_msg.header = header
            gaze_msg.width = 1080
            gaze_msg.height = 1920
            gaze_msg.u_2d = 0
            gaze_msg.v_2d = 0
            print('gaze point:',(0,0))
            gaze_pub.publish(gaze_msg)
        rate.sleep()
    except Exception as e:
        print(f"Error occurred: {e}")
        tobiiglasses.stop_streaming()
        tobiiglasses.close()
        print('close tobii glass successfully')


        

