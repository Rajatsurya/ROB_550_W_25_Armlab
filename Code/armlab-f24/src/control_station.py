#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
from functools import partial

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())
        self.ui.btn_calibrate.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))

        # User Buttons
        # TODO: Add more lines here to add more buttons
        # To make a button activate a state, copy the lines for btnUser3 but change 'execute' to whichever state you want
        self.ui.btnUser1.setText('Open Gripper')
        self.ui.btnUser1.clicked.connect(lambda: self.rxarm.gripper.release(delay=0.3))
        self.ui.btnUser2.setText('Close Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.gripper.grasp())
        self.ui.btnUser3.setText('Execute')
        self.ui.btnUser3.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser4.setText('Record')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'record'))
        self.ui.btnUser5.setText('Replay')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'replay')) 
        self.ui.btnUser6.setText('Clear Record')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'clear')) 
        self.ui.btnUser7.setText('Click2touch')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'click2touch'))
        self.ui.btnUser8.setText('save_img')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'save_img'))
        self.ui.btnUser9.setText('Click2reach')
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_init, 'click2reach'))      
        self.ui.btnUser10.setText('Sort')
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'sort'))
        self.ui.btnUser11.setText('Line em up')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'line'))
        self.ui.btnUser12.setText('To the sky')
        self.ui.btnUser12.clicked.connect(partial(nxt_if_arm_init, 'sky'))

        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    # Distances should be in mm
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad (roll)" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f rad (pitch)" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f rad (yaw)" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        # TODO: Modify this function to change the mouseover text.
        # You should make the mouseover text display the (x, y, z) coordinates of the pixel being hovered over

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            x, y = pt.x(), pt.y()
                # Ensure that homogenous_transform has been calculated
            if hasattr(self.camera, 'homography') and self.camera.homography is not None and self.camera.camera_calibrated:
                homography_inv = np.linalg.inv(self.camera.homography)
                pt_raw = np.array([pt.x(), pt.y(), 1])
                pt_raw = np.dot(homography_inv, pt_raw)
                pt_raw = pt_raw / pt_raw[2]
                x, y = int(pt_raw[0]), int(pt_raw[1])

            z = self.camera.DepthFrameRaw[y][x]
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                                (pt.x(), pt.y(), z))
            # self.ui.rdoutMouseWorld.setText("(-,-,-)")
            #changes have been made 
            world_pos = self.camera.coord_pixel_to_world(x, y, z)
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                                (world_pos[0], world_pos[1], world_pos[2]))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        # self.camera.last_click[0] = pt.x()
        # self.camera.last_click[1] = pt.y()
        # self.camera.new_click = True

        if self.camera.DepthFrameRaw.any() != 0:
            self.camera.last_click[0] = pt.x()
            self.camera.last_click[1] = pt.y()
            self.camera.new_click = True
            x, y = pt.x(), pt.y()
                # Ensure that homogenous_transform has been calculated
            if hasattr(self.camera, 'homography') and self.camera.homography is not None and self.camera.camera_calibrated:
                homography_inv = np.linalg.inv(self.camera.homography)
                pt_raw = np.array([pt.x(), pt.y(), 1])
                pt_raw = np.dot(homography_inv, pt_raw)
                pt_raw = pt_raw / pt_raw[2]
                x, y = int(pt_raw[0]), int(pt_raw[1])

            z = self.camera.DepthFrameRaw[y][x]
            #changes have been made 
            world_pos = self.camera.coord_pixel_to_world(x, y, z)
            self.camera.last_click_world = world_pos

        

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main():
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui()
    app_window.show()

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
if __name__ == '__main__':
    main()
