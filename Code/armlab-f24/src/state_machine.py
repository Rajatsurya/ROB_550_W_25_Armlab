"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from utils import recover_homogenous_transform_pnp, recover_homogeneous_transform_svd_modified
from camera import Camera
from rxarm import RXArm
from kinematics import IK_pox_numeric, IK_with_fixed_base_yaw, IK_analytics
from utils import euler_to_transformation_matrix, build_waypoint, build_waypoint_task1, build_waypoint_task3

counter_place = 1000
class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm:RXArm, camera:Camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]

        self.record_wp = []
        self.record_gripper = []
        self.INTERVAL = 0.2

        self.tags_world_xyz = {
            2: [250, -25, 0], 
            3: [250, 275, 0],
            4: [-250, 275, 0],
            6: [-250, -25, 0],
        }

        # self.dest_pts = {
        #     2: [1090, 700],
        #     3: [1090, 90], 
        #     4: [226, 90], 
        #     6: [226, 700],  
        # }
        self.dest_pts = {
            2: [915, 550],
            3: [915, 220], 
            4: [365, 220], 
            6: [365, 550],  
        }

        self.click2grab_xyz = None
        self.click2place_xyz = None
        self.waypoints_grab = []
        self.waypoints_place = []

        self.next_state_after_grab = None
        self.next_state_after_place = None

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.
        # print(self.camera.tag_detections)

        # print(f"Next state: {self.next_state}")

        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record":
            self.record()

        if self.next_state == "replay":
            self.replay()
        
        if self.next_state == "clear":
            self.clear()
        
        if self.next_state == "click2touch":
            self.click2touch()

        if self.next_state == "wait_for_click2grab":
            self.wait_for_click2grab()
        
        if self.next_state == "grab":
            if self.next_state_after_grab is not None:
                self.grab(self.next_state_after_grab)
            else:
                self.grab()

        if self.next_state == "wait_for_click2place":
            self.wait_for_click2place()

        if self.next_state == "place":
            if self.next_state_after_place is not None:
                self.place(self.next_state_after_place)
            else:
                self.place()

        if self.next_state == "save_img":
            self.save_img()        

        if self.next_state == "click2reach":
            self.click2reach()

        if self.next_state == "wait_for_click2reach":
            self.wait_for_click2reach()
        
        if self.next_state == "reach":
            self.reach()

        if self.next_state == "sort":
            self.sort()
        
        if self.next_state == "sort_manager":
            self.sort_manager()

        if self.next_state == "line":
            self.line()

        if self.next_state == "line_manager":
            self.line_manager()

        if self.next_state == "sky":
            self.sky()
        
        if self.next_state == "sky_manager":
            self.sky_manager()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """

        # If no current process is stored, initialize it
        if not hasattr(self, "current_waypoint_index"):
            self.current_waypoint_index = 0
            self.current_step_start_time = time.time()

        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.waypoints):
            self.status_message = "State: Execute - All waypoints executed"
            self.next_state = "idle"
            del self.current_waypoint_index  # Clean up temporary state variables
            del self.current_step_start_time
            return

        # Get the target position for the current waypoint
        target_position = np.array(self.waypoints[self.current_waypoint_index])

        # Get the current positions of the arm
        current_positions = np.array(self.rxarm.get_positions())

        # Calculate the positional difference
        position_diff = np.abs(current_positions - target_position)

        # Check if the target position is reached within the tolerance
        if np.all(position_diff <= 0.1):
            # Move to the next waypoint
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()  # Reset the start time for the next step
            return

        # Send the command to set the target positions
        self.rxarm.set_positions(target_position)

        # Check if the operation is taking too long
        if time.time() - self.current_step_start_time > 2:
            print(f"Timeout reached while moving to position: {target_position}")
            # Handle timeout (e.g., skip to the next waypoint or transition to an error state)
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()

        # Update the status message
        self.status_message = (
            f"State: Execute - Moving to waypoint {self.current_waypoint_index + 1}/"
            f"{len(self.waypoints)}"
    )

        

    def record(self):
        """
        @brief record the position every self.Interval time, press 'q' to stop recording
        """
        self.current_state = "record"
        self.rxarm.disable_torque()
        time.sleep(0.1)
        # self.record_wp = []
        # key = cv2.waitKey(1) & 0xFF
        # while(True):
        #     self.record_wp.append(self.rxarm.get_positions())
        #     time.sleep(self.INTERVAL)
        #     if key == ord("q"):
        #         break
        self.record_wp.append(self.rxarm.get_positions())
        # self.record_gripper.append(self.rxarm.get_gripper_state())
        self.next_state = "idle"

    def replay(self):
            
        """!
        @brief      Go through all waypoints, replaying the plan 5 times
        TODO: Implement this function to execute a waypoint plan
            Make sure you respect estop signal
        """

        self.rxarm.enable_torque()
        # If no replay state is stored, initialize it
        if not hasattr(self, "replay_count"):
            self.replay_count = 0

        # If replay count exceeds 5, stop execution
        if self.replay_count >= 5:
            self.status_message = "State: Execute - Replay complete"
            self.next_state = "idle"
            del self.replay_count
            if hasattr(self, "current_waypoint_index"):
                del self.current_waypoint_index
                del self.current_step_start_time
            return

        # If no current process is stored, initialize it
        if not hasattr(self, "current_waypoint_index"):
            self.current_waypoint_index = 0
            self.current_step_start_time = time.time()

        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.record_wp):
            self.status_message = (
                f"State: Execute - Replay {self.replay_count + 1}/5 completed"
            )
            self.current_waypoint_index = 0  # Reset for the next replay
            self.replay_count += 1
            return

        # Get the target position for the current waypoint
        target_position = np.array(self.record_wp[self.current_waypoint_index])

        # Get the current positions of the arm
        current_positions = np.array(self.rxarm.get_positions())

        # Calculate the positional difference
        position_diff = np.abs(current_positions - target_position)

        # Check if the target position is reached within the tolerance
        if np.all(position_diff <= 0.1):
            # Move to the next waypoint
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()  # Reset the start time for the next step
            return

        # Send the command to set the target positions
        self.rxarm.set_positions(target_position)

        # Check if the operation is taking too long
        if time.time() - self.current_step_start_time > 2:
            print(f"Timeout reached while moving to position: {target_position}")
            # Handle timeout (e.g., skip to the next waypoint or transition to an error state)
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()

        # Update the status message
        self.status_message = (
            f"State: Execute - Moving to waypoint {self.current_waypoint_index + 1}/"
            f"{len(self.record_wp)} (Replay {self.replay_count + 1}/5)"
        )


    
    def clear(self):
        self.current_state = "clear"
        self.next_state = "idle"
        self.record_wp = []
        

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"
        print("Starting calibration...")
        # Perform calibration routine
        msg = self.camera.tag_detections
        if msg is None:
            print("No tags detected")
            return
        # Get the tag detection message
        tags_pixel_xy = {}
        tags_pixel_xyz = {}
        tags_world_xyz = self.tags_world_xyz

        def pixel_to_camera_xyz(u, v, d, K):
            """
            Convert a pixel coordinate + depth to 3D camera coordinates.
            Args:
                u, v: Pixel coordinates (floats or ints).
                d: Depth (the same distance unit as you want in 3D).
                K: Intrinsic matrix, 3x3.
            Returns:
                (X, Y, Z) in camera frame
            """
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d
            return (X, Y, Z)

        for detection in msg.detections:
            center = (int(detection.centre.x), int(detection.centre.y))
            z = self.camera.DepthFrameRaw[center[1], center[0]]
            center_xyz = (center[0], center[1], z)
            tags_pixel_xy[int(detection.id)] = center
            tags_pixel_xyz[int(detection.id)] = pixel_to_camera_xyz(center[0], center[1], z, self.camera.intrinsic_matrix)

        # Ensure we have common tags between detected and world points
        common_tags = set(tags_pixel_xy.keys()) & set(tags_world_xyz.keys())
        if len(common_tags) < 4:  # SolvePnP requires at least 4 correspondences
            print("Not enough tags for calibration (need at least 4)")
            print(tags_pixel_xy)
            print(tags_world_xyz)
            return

        # Prepare image points and world points
        image_points_xy = np.array([tags_pixel_xy[tag_id] for tag_id in common_tags], dtype=np.float32)
        image_points_xyz = np.array([tags_pixel_xyz[tag_id] for tag_id in common_tags], dtype=np.float32)

        world_points = np.array([tags_world_xyz[tag_id] for tag_id in common_tags], dtype=np.float32)

        # Camera intrinsics matrix (K) and distortion coefficients (D)
        K = self.camera.intrinsic_matrix  # Assuming intrinsics are stored in camera object
        D = self.camera.dist_coeffs  # Assuming distortion coefficients are stored in camera object
        # D = None

        # Recover the homogenous transformation using the function
        try:
            # homogenous_transform = recover_homogenous_transform_pnp(image_points_xy, world_points, K, D)
            homogenous_transform = recover_homogeneous_transform_svd_modified(world_points, image_points_xyz)
            self.camera.extrinsic_matrix = homogenous_transform[:3, :]

            self.camera.homogenous_transform = homogenous_transform
            self.camera.camera_calibrated = True
            print("Calibration successful!")
            print("Homogenous transformation matrix:")
            print(homogenous_transform)
        except Exception as e:
            print(f"Error during calibration: {e}")

        
        # calibrate color image using hoomography
        # get the destination points
        common_tags = set(tags_pixel_xy.keys()) & set(self.dest_pts.keys())
        src_pts = np.array([tags_pixel_xy[tag_id] for tag_id in common_tags], dtype=np.float32)
        dst_pts = np.array([self.dest_pts[tag_id] for tag_id in common_tags], dtype=np.float32)
        h, status = cv2.findHomography(src_pts, dst_pts)
        self.camera.homography = h
        print("Homography matrix:")
        print(h)
        print("Calibration complete")
        return

    def click2touch(self):
        if not self.camera.camera_calibrated:
            print("Camera not calibrated")
            self.next_state = "idle"
            return
        self.camera.new_click = False
        self.next_state = "wait_for_click2grab"
        return
    
    def wait_for_click2grab(self):
        print("Waiting for click to grab")
        if self.camera.new_click:
            self.click2grab_xyz = self.camera.last_click_world
            print(f"Click to grab at {self.click2grab_xyz}")
            # calculate way points:
            x,y,z = self.click2grab_xyz
            z = max(z,0) / 9 * 10 - 15 
            # error from realsense
            # do gravity compensation, linearly depend on the distance to the center
            # radius = np.sqrt(x**2 + y**2)
            # compensation_z = - 0.5668 * radius + 123.79
            # z = z - compensation_z
            # safe consideration
            z = max(z, 10)

            T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
            # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
            q_desired = IK_analytics(T_desired)[0]
            # compensate for the offset
            # q_desired[2] = q_desired[2] - 3 / 180 * np.pi
            wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
            wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+50, 0, 0, 0))[0]
            self.waypoints_grab = [wp1, wp2, q_desired]
            self.rxarm.gripper.release()
            self.next_state = "grab"
            if hasattr(self, "current_waypoint_index"):
                del self.current_waypoint_index  # Clean up temporary state variables
            if hasattr(self, "current_step_start_time"):
                del self.current_step_start_time
        else:
            self.next_state = "wait_for_click2grab"
        return
    
    def grab(self, next_state="wait_for_click2place"):
        self.next_state_after_grab = next_state
        if self.click2grab_xyz is None:
            print("No click to grab point")
            self.next_state = "idle"
            return
        
        # move to the click to grab point

        # If no current process is stored, initialize it
        if not hasattr(self, "current_waypoint_index"):
            self.current_waypoint_index = 0
            self.current_step_start_time = time.time()

        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.waypoints_grab):
            self.status_message = "State: Grab - reached click to grab point"
            print("Reached click to grab point")
            print("Next state: ", next_state)
            self.next_state = next_state
            del self.current_waypoint_index  # Clean up temporary state variables
            del self.current_step_start_time
            time.sleep(1)
            self.rxarm.gripper.grasp() # grasp the object
            time.sleep(1.5)
            self.camera.new_click = False
            self.waypoints_grab = []
            self.click2grab_xyz = None
            self.next_state_after_grab = None
            return

        # Get the target position for the current waypoint
        target_position = np.array(self.waypoints_grab[self.current_waypoint_index])

        # Get the current positions of the arm
        current_positions = np.array(self.rxarm.get_positions())

        # Calculate the positional difference
        position_diff = np.abs(current_positions - target_position)

        # Check if the target position is reached within the tolerance
        if np.all(position_diff <= 0.15):
            # Move to the next waypoint
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()  # Reset the start time for the next step
            return

        # Send the command to set the target positions
        # self.rxarm.gripper.release()
        time.sleep(0.25)
        self.rxarm.set_positions(target_position)

        # Check if the operation is taking too long
        if time.time() - self.current_step_start_time > 5:
            print(f"Timeout reached while moving to position: {target_position}")
            print(f"Current position: {current_positions}")
            # Handle timeout (e.g., skip to the next waypoint or transition to an error state)
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()

        # Update the status message
        self.status_message = (
            f"State: Execute - Moving to waypoint {self.current_waypoint_index + 1}/"
            f"{len(self.waypoints_grab)}"
        )
        
        self.next_state = "grab"
        return

    def wait_for_click2place(self):
        print("Waiting for click to place")
        if self.camera.new_click:
            self.click2place_xyz = self.camera.last_click_world
            # calculate way points:
            x,y,z = self.click2place_xyz
            z = max(z/9*10, 20) + 10
            print(f"Click to place at {x,y,z}")
            T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
            # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
            q_desired = IK_analytics(T_desired)[0]

            wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
            wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+90, 0, 0, 0))[0]
            self.waypoints_place = [wp1, wp2, q_desired]
            if any(np.isnan(q_desired)) or max(np.abs(q_desired[:4])) > np.pi/2:
                print("No solution found for click to place point")
                self.camera.new_click = False
                self.next_state = "wait_for_click2place"
                return
            self.next_state = "place"
            if hasattr(self, "current_waypoint_index"):
                del self.current_waypoint_index  # Clean up temporary state variables
            if hasattr(self, "current_step_start_time"):
                del self.current_step_start_time
        else:
            self.next_state = "wait_for_click2place"
        return

    def place(self, next_state="idle"):
        self.next_state_after_place = next_state
        if self.click2place_xyz is None:
            print("No click to place point")
            self.next_state = next_state
            return
        
        # move to the click to place point

        # If no current process is stored, initialize it
        if not hasattr(self, "current_waypoint_index"):
            self.block_placed = False
            self.current_waypoint_index = 0
            self.current_step_start_time = time.time()


        if len(self.waypoints_place) == 4 and not self.block_placed and self.current_waypoint_index == len(self.waypoints_place) - 1:
                self.rxarm.gripper.release()
                time.sleep(1)
                self.block_placed = True

        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.waypoints_place):

            time.sleep(1)
            self.rxarm.gripper.release() # release the object
            time.sleep(1)

            self.status_message = "State: place - reached click to place point"

            print("Reached click to place point")
            print("Next state: ", next_state)
            self.next_state = next_state
            del self.current_waypoint_index  # Clean up temporary state variables
            del self.current_step_start_time
            del self.block_placed
            # self.rxarm.gripper.release() # release the object
            self.camera.new_click = False
            self.waypoints_place = []
            self.click2place_xyz = None
            self.next_state_after_place = None
            return
        
        
        # Get the target position for the current waypoint
        target_position = np.array(self.waypoints_place[self.current_waypoint_index])

        # Get the current positions of the arm
        current_positions = np.array(self.rxarm.get_positions())

        # Calculate the positional difference
        position_diff = np.abs(current_positions - target_position)

        # Check if the target position is reached within the tolerance
        if np.all(position_diff <= 0.02):
            # Move to the next waypoint
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()  # Reset the start time for the next step
            return

        # Send the command to set the target positions
        # self.rxarm.gripper.grasp()
        self.rxarm.set_positions(target_position)

        # Check if the operation is taking too long
        if time.time() - self.current_step_start_time > 6:
            print(f"Timeout reached while moving to position: {target_position}")
            # Handle timeout (e.g., skip to the next waypoint or transition to an error state)
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()

        # Update the status message
        self.status_message = (
            f"State: Execute - Moving to waypoint {self.current_waypoint_index + 1}/"
            f"{len(self.waypoints_place)}"
        )
        
        self.next_state = "place"
        return
    
    def save_img(self):
        self.camera.save_img()
        self.next_state = "idle"

    def click2reach(self):
        if not self.camera.camera_calibrated:
            print("Camera not calibrated")
            self.next_state = "idle"
            return
        self.camera.new_click = False
        self.next_state = "wait_for_click2reach"
        return

    def wait_for_click2reach(self):
        print("Waiting for click to reach")
        if self.camera.new_click:
            self.click2reach_xyz = self.camera.last_click_world
            # calculate way points:
            x,y,z = self.click2reach_xyz
            z = max(z,0) / 9 * 10 # error from realsense
            # do gravity compensation, linearly depend on the distance to the center
            radius = np.sqrt(x**2 + y**2)
            # safe consideration
            z = max(z, 10)

            print(f"Click to reach at {x,y,z}")
            T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
            # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
            q_desired = IK_analytics(T_desired)[0]

            wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
            wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+50, 0, 0, 0))[0]
            self.waypoints_reach = [wp1, wp2, q_desired]
            # if any(np.isnan(q_desired)) or max(np.abs(q_desired[:4])) > np.pi/2 * 1.3:
            #     print("No solution found for click to reach point")
            #     self.camera.new_click = False
            #     self.next_state = "wait_for_click2reach"
            #     return
            self.next_state = "reach"
            if hasattr(self, "current_waypoint_index"):
                del self.current_waypoint_index  # Clean up temporary state variables
            if hasattr(self, "current_step_start_time"):
                del self.current_step_start_time
        else:
            self.next_state = "wait_for_click2reach"
        return
    


    def reach(self, next_state="idle"):
        if self.click2reach_xyz is None:
            print("No click to reach point")
            self.next_state = next_state
            return
        
        # move to the click to reach point

        # If no current process is stored, initialize it
        if not hasattr(self, "current_waypoint_index"):
            self.current_waypoint_index = 0
            self.current_step_start_time = time.time()

        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.waypoints_reach):
            self.status_message = "State: reach - reached click to reach point"
            self.next_state = next_state
            del self.current_waypoint_index  # Clean up temporary state variables
            del self.current_step_start_time
            self.rxarm.gripper.release() # release the object
            self.camera.new_click = False
            self.waypoints_reach = []
            return

        # Get the target position for the current waypoint
        target_position = np.array(self.waypoints_reach[self.current_waypoint_index])

        # Get the current positions of the arm
        current_positions = np.array(self.rxarm.get_positions())

        # Calculate the positional difference
        position_diff = np.abs(current_positions - target_position)

        # Check if the target position is reached within the tolerance
        if np.all(position_diff <= 0.1):
            # Move to the next waypoint``
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()  # Reset the start time for the next step
            return

        # Send the command to set the target positions
        self.rxarm.gripper.grasp()
        self.rxarm.set_positions(target_position)

        # Check if the operation is taking too long
        if time.time() - self.current_step_start_time > 10:
            print(f"Timeout reached while moving to position: {target_position}")
            # Handle timeout (e.g., skip to the next waypoint or transition to an error state)
            self.current_waypoint_index += 1
            self.current_step_start_time = time.time()

        # Update the status message
        self.status_message = (
            f"State: Execute - Moving to waypoint {self.current_waypoint_index + 1}/"
            f"{len(self.waypoints_reach)}"
        )
        
        self.next_state = "reach"       


    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.detected_blocks = self.camera.blockDetector()
        

    def sort(self):
        self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=0.5,
                              blocking=False)
        self.rxarm.gripper.release()
        time.sleep(3.5)
        detected_blocks = self.camera.blockDetector(lower=10, upper=120)

        world_xyz_all = [blocks[2] for blocks in detected_blocks]
        world_xyz_all = np.array(world_xyz_all)

        rainbow_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']

        # Sort detected_blocks by the predefined rainbow order
        detected_blocks = sorted(detected_blocks, key=lambda x: rainbow_order.index(x[0]))

        self.cp1_operation_list = []
        
        small_num = 0
        large_num = 0
        
        for res in detected_blocks:
            color, theta, world_xyz, type = res
            x,y,z = world_xyz
            if y < 0: continue # already sorted
            if np.sqrt(x**2 + y**2) > 550: 
                continue # too close/far to the center
            if type == "small":
                dest_xyz = (-200, -100, small_num*25)
                small_num += 1
            elif type == "large":
                dest_xyz = (200, -100, large_num*40)
                large_num += 1
            
            self.cp1_operation_list.append((res, dest_xyz))

        # self.cp1_operation_list.sort(key=lambda x: x[0][2][0]**2 + x[0][2][1]**2)
            
        for res, dest_xyz in self.cp1_operation_list:
            print(f"Color: {color}, Type: {type}, World XYZ: {world_xyz}, Dest XYZ: {dest_xyz}")
        
        self.next_state = "sort_manager"
        self.click2grab_xyz  = None
        self.click2place_xyz = None
    
    
    def sort_manager(self):
        if self.click2grab_xyz is not None or self.click2place_xyz is not None:
            self.next_state = "place"
            self.place("sort_manager")
            return
        if len(self.cp1_operation_list) == 0:
            self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                        accel_time=0.5,
                        blocking=True)
            self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                                accel_time=0.5,
                                blocking=False)
            self.rxarm.gripper.release()
            time.sleep(3.5)
            detected_blocks = self.camera.blockDetector(lower=10, upper=120)
            
            

            for res in detected_blocks:
                color, theta, world_xyz, type = res
                x,y,z = world_xyz
                if y < 0: continue # already sorted
                if np.sqrt(x**2 + y**2) > 350: 
                    continue # too close/far to the center
                if type == "small":
                    dest_xyz = (-150, -150, 30)
                elif type == "large":
                    dest_xyz = (150, -150, 30)
            
                self.cp1_operation_list.append((res, dest_xyz))
            
            self.cp1_operation_list.sort(key=lambda x: x[0][2][0]**2 + x[0][2][1]**2)

            if len(self.cp1_operation_list) == 0:
                self.next_state = "idle"
            else:
                self.next_state = "sort_manager"
                self.click2grab_xyz  = None
                self.click2place_xyz = None
                
            return
        else:
            res, dest_xyz = self.cp1_operation_list.pop(0)
            color, theta, world_xyz, type = res
            self.click2grab_xyz, self.click2place_xyz = world_xyz, dest_xyz
            self.waypoints_grab = build_waypoint_task1(self.click2grab_xyz, theta=theta, type="grab")
            self.waypoints_place = build_waypoint_task1(self.click2place_xyz, type="place")
            self.rxarm.gripper.grasp()
            self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
            self.rxarm.gripper.release()
            self.next_state = "grab"
            self.grab("sort_manager")
        
            return
        

    def line(self):
        self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=0.5,
                              blocking=False)
        self.rxarm.gripper.release()
        time.sleep(3.5)
        detected_blocks = self.camera.blockDetector(lower=10, upper=120)

        rainbow_order = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']

        # Sort detected_blocks by the predefined rainbow order
        detected_blocks = sorted(detected_blocks, key=lambda x: rainbow_order.index(x[0]))

        self.cp1_operation_list = []
        
        small_num = 0
        large_num = 0
        
        for res in detected_blocks:
            color, theta, world_xyz, type = res
            x,y,z = world_xyz
            if y < 0: continue # already sorted
            if np.sqrt(x**2 + y**2) > 550: 
                continue # too close/far to the center
            if type == "small":
                dest_xyz = (-250, 150-25*small_num, 10)
                small_num += 1
            elif type == "large":
                dest_xyz = (250, 175-40*large_num, 15)
                large_num += 1
            
            self.cp1_operation_list.append((res, dest_xyz))

        # self.cp1_operation_list.sort(key=lambda x: x[0][2][0]**2 + x[0][2][1]**2)
            
        for res, dest_xyz in self.cp1_operation_list:
            print(f"Color: {color}, Type: {type}, World XYZ: {world_xyz}, Dest XYZ: {dest_xyz}")
        
        self.next_state = "line_manager"
        self.click2grab_xyz  = None
        self.click2place_xyz = None

    def line_manager(self):
        if self.click2grab_xyz is not None or self.click2place_xyz is not None:
            self.next_state = "place"
            self.place("sort_manager")
            return
        if len(self.cp1_operation_list) == 0:
            self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                        accel_time=0.5,
                        blocking=True)
            self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                                accel_time=0.5,
                                blocking=False)
            self.rxarm.gripper.release()
            time.sleep(3.5)
            detected_blocks = self.camera.blockDetector(lower=10, upper=120)
            
            for res in detected_blocks:
                color, theta, world_xyz, type = res
                x,y,z = world_xyz
                if y < 0: continue # already sorted
                if np.sqrt(x**2 + y**2) > 350: 
                    continue # too close/far to the center
                if type == "small":
                    dest_xyz = (-150, -150, 30)
                elif type == "large":
                    dest_xyz = (150, -150, 30)
            
                self.cp1_operation_list.append((res, dest_xyz))
            
            self.cp1_operation_list.sort(key=lambda x: x[0][2][0]**2 + x[0][2][1]**2)

            if len(self.cp1_operation_list) == 0:
                self.next_state = "idle"
            else:
                self.next_state = "sort_manager"
                self.click2grab_xyz  = None
                self.click2place_xyz = None
                
            return
        else:
            res, dest_xyz = self.cp1_operation_list.pop(0)
            color, theta, world_xyz, type = res
            self.click2grab_xyz, self.click2place_xyz = world_xyz, dest_xyz
            self.waypoints_grab = build_waypoint_task1(self.click2grab_xyz, theta=theta, type="grab")
            self.waypoints_place = build_waypoint_task1(self.click2place_xyz, type="place")
            self.rxarm.gripper.grasp()
            self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
            self.rxarm.gripper.release()
            self.next_state = "grab"
            self.grab("sort_manager")
        
            return
        

    def sky(self):
        self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
        self.rxarm.arm.go_to_sleep_pose(moving_time=2,
                              accel_time=0.5,
                              blocking=False)
        self.rxarm.gripper.release()
        time.sleep(3.5)
        self.cp1_operation_list = []
        
        for i in range(7):
            res = ("red", 0, (0, 250, 40), "big")
            if i == 0:
                dest_xyz = (195, 0, 40 * i)
            else:
                dest_xyz = (200, 0, 40 * i+1)
            self.cp1_operation_list.append((res, dest_xyz))

        # self.cp1_operation_list.sky(key=lambda x: x[0][2][0]**2 + x[0][2][1]**2)
            
        self.next_state = "sky_manager"
        self.click2grab_xyz  = None
        self.click2place_xyz = None
    
    
    def sky_manager(self):
        if self.click2grab_xyz is not None or self.click2place_xyz is not None:
            self.next_state = "place"
            self.place("sky_manager")
            return
        if len(self.cp1_operation_list) == 0:

            if len(self.cp1_operation_list) == 0:
                self.next_state = "idle"
            else:
                self.next_state = "sky_manager"
                self.click2grab_xyz  = None
                self.click2place_xyz = None
                
            return
        else:
            res, dest_xyz = self.cp1_operation_list.pop(0)
            color, theta, world_xyz, type = res
            self.click2grab_xyz, self.click2place_xyz = world_xyz, dest_xyz
            self.waypoints_grab = build_waypoint_task3(self.click2grab_xyz, theta=theta, type="grab")
            self.waypoints_place = build_waypoint_task3(self.click2place_xyz, theta = 90, type="place")
            self.rxarm.gripper.grasp()
            self.rxarm.arm.go_to_home_pose(moving_time=1.5,
                             accel_time=0.5,
                             blocking=True)
            self.rxarm.gripper.release()
            self.next_state = "grab"
            self.grab("sky_manager")
        
            return
        

    

       



    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)