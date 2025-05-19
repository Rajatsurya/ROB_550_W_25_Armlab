#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from utils import detection


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        # self.intrinsic_matrix = np.eye(3)
        self.intrinsic_matrix = np.array([908.61087, 0.0, 642.62399, 
                                            0.0, 908.59074, 358.7872, 
                                            0.0, 0.0, 1.0], dtype=np.float64).reshape((3, 3)) #this is to test it at the moment
        
        self.dist_coeffs = np.array([0.135471, -0.219797, 0.002950, 0.004832, 0.000000], dtype=np.float64)
        # self.extrinsic_matrix = np.eye(4)

        degree_x = 187.5#188
        cos_x = np.cos(degree_x / 180 * np.pi)
        sin_x = np.sin(degree_x / 180 * np.pi)
        
        degree_y = 0.2#1.4
        cos_y = np.cos(degree_y / 180 * np.pi)
        sin_y = np.sin(degree_y / 180 * np.pi)

        self.extrinsic_matrix = np.array([
            [cos_y,    sin_x * sin_y,  cos_x*sin_y, 18 ], 
            [0.,    cos_x,  -sin_x, 259-115+20], 
            [-sin_y,  sin_x * cos_y, cos_x * cos_y, 992+21], 
        ], dtype=np.float64)

        self.homogenous_transform = None
        self.homography = None

        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.last_click_world = np.array([0, 0, 0]) # This contains the last clicked position in world coordinates
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.detected_blocks = []

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self, lower=10, upper=150):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        if not self.camera_calibrated or self.homogenous_transform is None or self.homogenous_transform is None:
            print("Camera not calibrated. Block detecting unable to proceed.")
            return
        self.VideoFrame, self.detected_blocks = detection(
            self.VideoFrame, # rgb_image with homography applied
            self.VideoFrame, # cnt_image with homography applied same as rgb_image for simplify
            self.DepthFrameRaw, 
            self.homogenous_transform, 
            self.intrinsic_matrix, 
            self.homography,
            lower=lower, 
            upper=upper
        )
        return self.detected_blocks

        # detected_blocks = [ (color, theta, world_xyz, ['large'/'small']) ]


    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        if self.camera_calibrated == True:
            modified_image = self.VideoFrame.copy()
            # Draw Grid of Points
            X, Y = np.meshgrid(self.grid_x_points, self.grid_y_points)
            grid_points = np.vstack((X.ravel(), Y.ravel(), (np.ones_like(X).ravel()), np.ones_like(X).ravel())).astype(np.float32)

            camera_points = np.dot(self.extrinsic_matrix[0:3, :], grid_points)
            pixel_points = np.dot(self.intrinsic_matrix, camera_points)
            pixel_x = pixel_points[0, :] / pixel_points[2, :]
            pixel_y = pixel_points[1, :] / pixel_points[2, :]
            original_points = np.vstack((pixel_x, pixel_y, np.ones_like(pixel_x)))
            transformed_points = self.homography @ original_points
            transformed_points /= transformed_points[2]  # Normalize to homogeneous coordinates
            # Draw the transformed points on the image
            for i in range(transformed_points.shape[1]):
                point_pos = (int(transformed_points[0, i]), int(transformed_points[1, i]))
                cv2.circle(modified_image, point_pos, 3, (0, 255, 0), thickness=-1)
            

            # Find the pixel coordinates for the point (0, 0) in the grid
            zero_grid_point = np.array([[0], [0], [1], [1]], dtype=np.float32)
            zero_camera_point = np.dot(self.extrinsic_matrix[0:3, :], zero_grid_point)
            zero_pixel_point = np.dot(self.intrinsic_matrix, zero_camera_point)
            zero_pixel_x = zero_pixel_point[0, 0] / zero_pixel_point[2, 0]
            zero_pixel_y = zero_pixel_point[1, 0] / zero_pixel_point[2, 0]
            transformed_zero_point = self.homography @ np.array([[zero_pixel_x], [zero_pixel_y], [1]])
            zero_pixel_x = transformed_zero_point[0] / transformed_zero_point[2]
            zero_pixel_y = transformed_zero_point[1] / transformed_zero_point[2]
            # Draw the point (0, 0) with a red circle of radius 5
            zero_point_pos = (int(zero_pixel_x), int(zero_pixel_y))
            cv2.circle(modified_image, zero_point_pos, 6, (255, 0, 0), thickness=-1)

        # Write your code here

        self.GridFrame = modified_image

     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        for detection in msg.detections:
            center = (int(detection.centre.x), int(detection.centre.y))
            cv2.circle(modified_image, center, 5, (0, 0, 255), -1)
            for corner in detection.corners:
                cv2.circle(modified_image, (int(corner.x), int(corner.y)), 5, (0, 255, 0), -1)
            cv2.putText(modified_image, str(detection.id), (center[0]-10, center[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        self.TagImageFrame = modified_image

#made this change
    def coord_pixel_to_world(self, u, v, depth):
        if self.camera_calibrated and self.homogenous_transform is not None:
            # Step 1: Back-project the pixel coordinate into the camera coordinate system
            K_inv = np.linalg.inv(self.intrinsic_matrix)  # Inverse of camera intrinsics matrix
            pixel_homogeneous = np.array([u, v, 1])  # Homogeneous pixel coordinate
            camera_point = depth * K_inv @ pixel_homogeneous  # 3D point in the camera coordinate system

            # Step 2: Transform the point from camera coordinates to world coordinates
            # Convert to homogeneous 3D coordinates in the camera frame
            camera_homogeneous = np.append(camera_point, 1)  # [x, y, z, 1]

            # Apply the inverse transformation to map to world coordinates
            world_point_homogeneous = np.linalg.inv(self.homogenous_transform) @ camera_homogeneous

            # Step 3: Extract the world coordinates
            world_point = world_point_homogeneous[:3] / world_point_homogeneous[3]  # Normalize to Cartesian
            return world_point
        else:
            pixel_coord = (u, v)
            extrinsic_matrix = self.extrinsic_matrix
            intrinsic_matrix = self.intrinsic_matrix
            # Extract rotation (R) and translation (t) from extrinsic matrix
            R = extrinsic_matrix[:3, :3]
            t = extrinsic_matrix[:3, 3]
            
            # Invert rotation matrix
            R_inv = np.linalg.inv(R)
            
            # Convert pixel to normalized image coordinates
            u, v = pixel_coord
            pixel_homogeneous = np.array([u, v, 1])  # Homogeneous pixel coordinates
            
            # Compute normalized image coordinates
            normalized_coord = np.linalg.inv(intrinsic_matrix) @ pixel_homogeneous
            
            # Scale by depth (Z)
            camera_coord = depth * normalized_coord  # Camera space coordinates
            
            # Convert to world coordinates
            world_coord = R_inv @ (camera_coord - t)
            
            return world_coord

#this is to convert the camera frame to world frame

    def save_img(self, path=None):
        if path is None:
            img_path = "output.png"
            depth_path = "output_depth.png"
        #cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
        video_frame = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, video_frame)
        # save depth frame raw
        cv2.imwrite(depth_path, self.DepthFrameRaw)



class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera:Camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        
        # Ensure that homogenous_transform has been calculated
        if not hasattr(self.camera, 'homography') or self.camera.homography is None or not self.camera.camera_calibrated:
            # print("Homography not calculated. Unable to proceed.")
            return

        height, width = cv_image.shape[:2]
        transformed_image = cv2.warpPerspective(cv_image, self.camera.homography, (width, height))
        self.camera.VideoFrame = transformed_image

        # # Retrieve the precomputed homogenous transformation matrix
        # H = self.camera.homogenous_transform

        # # Step 1: Warp the image using the homogenous transform
        # height, width = cv_image.shape[:2]
        # transformed_image = cv2.warpPerspective(cv_image, H, (width, height))

        # self.camera.VideoFrame = transformed_image
        
        # draw detection
        self.camera.blockDetector()


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()