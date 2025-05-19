import cv2
import numpy as np
from kinematics import IK_analytics


def recover_homogenous_transform_pnp(image_points, world_points, K, D):
    '''
    Use SolvePnP to find the rigidbody transform representing the camera pose in
    world coordinates (not working)
    '''
    distCoeffs = D
    [_, R_exp, t] = cv2.solvePnP(world_points,
                                 image_points,
                                 K,
                                 distCoeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(R_exp)
    return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

def recover_homogeneous_transform_svd(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))
    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))
    T = d_bar - np.dot(m_bar, R)
    return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))


def recover_homogeneous_transform_svd_modified(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    This modification recalculates R after finding T, which is technically incorrect but in practice works
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))

    # Get translation
    T = d_bar - np.dot(m_bar, R)

    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))

    T = np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))

    return T


def euler_to_transformation_matrix(x, y, z, roll, pitch, yaw):
    """
    Create 4x4 transformation matrix from translation (x,y,z) and Euler angles (roll, pitch, yaw).
    
    Args:
    x, y, z (float): Translation coordinates
    roll, pitch, yaw (float): Rotation angles in radians
    
    Returns:
    numpy.ndarray: 4x4 homogeneous transformation matrix
    """
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (XYZ order)
    R = Rz @ Ry @ Rx
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def build_waypoint_task1(target_xyz, theta=0, type=None, fast = False):
    """
    Create a waypoint for the robot to grab an object.
    
    Args:
    target_xyz: Translation coordinates
    theta (float): Rotation angle in radians
    
    Returns:
    list of np.ndarray: List of joint angles for the robot to reach the waypoint
    """
    x, y, z = target_xyz

    # Translation and rotation
    z = max(z,0) / 9 * 10 # error from realsense

    # compensation
    if type == "grab":
        z = z - 15
    elif type == "place":
        z = z + 10

    z = max(z, 10)
    print(f"Click to reach at {x,y,z}")
    T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
    # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
    q_desired, reach_type = IK_analytics(T_desired, theta=theta)
    if type == "place" and z < 160:
        q_desired, reach_type = IK_analytics(T_desired, theta=theta, type="H")
        reach_type = "V"
    else:
        q_desired, reach_type = IK_analytics(T_desired, theta=theta)

    if reach_type == "V":
        # gravity compensation
        r = np.sqrt(x**2 + y**2)
        cooef= np.array([-1.450000e-06,  5.735600e-04, -7.048614e-02,  6.217500e-02])
        compensation = cooef[0]*r**3 + cooef[1]*r**2 + cooef[2]*r + cooef[3]
        z = z - compensation

        wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
        wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+50, 0, 0, 0), theta=theta, type="V")[0]
        q_desired = IK_analytics(euler_to_transformation_matrix(x, y, z, 0, 0, 0), theta=theta, type="V")[0]

    else:
        # shink radius
        distar = 50
        x_dis = distar / np.sqrt(x**2 + y**2) * x
        y_dis = distar / np.sqrt(x**2 + y**2) * y

        wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
        # wp2 = IK_analytics(euler_to_transformation_matrix(x - x_dis, y - y_dis , z, 0, 0, 0), theta=theta, type="H")[0]
        wp2 = np.array([q_desired[0], q_desired[1] - 1, q_desired[2] - 1, q_desired[3] - 1, 0])

    if fast:
        waypoints = [wp1, q_desired]
    else:
        if type == "place":
            waypoints = [wp1, wp2, q_desired, wp2]
        else:
            waypoints = [wp1, wp2, q_desired]
    return waypoints

def build_waypoint_task3(target_xyz, theta=0, type=None, fast = False):
    """
    Create a waypoint for the robot to grab an object.
    
    Args:
    target_xyz: Translation coordinates
    theta (float): Rotation angle in radians
    
    Returns:
    list of np.ndarray: List of joint angles for the robot to reach the waypoint
    """
    x, y, z = target_xyz

    # Translation and rotation
    z = max(z,0) / 9 * 10 # error from realsense

    # compensation
    if type == "grab":
        z = z - 15
    elif type == "place":
        z = z + 15

    z = max(z, 10)
    print(f"Click to reach at {x,y,z}")
    T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
    # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
    q_desired, reach_type = IK_analytics(T_desired, theta=theta)
    if type == "place" and z < 160:
        q_desired, reach_type = IK_analytics(T_desired, theta=theta, type="H")
        reach_type = "V"
    else:
        q_desired, reach_type = IK_analytics(T_desired, theta=theta)

    if reach_type == "V":
        # gravity compensation
        r = np.sqrt(x**2 + y**2)
        cooef= np.array([-1.450000e-06,  5.735600e-04, -7.048614e-02,  6.217500e-02])
        compensation = cooef[0]*r**3 + cooef[1]*r**2 + cooef[2]*r + cooef[3]
        z = z - compensation

        wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
        wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+50, 0, 0, 0), theta=theta, type="V")[0]
        q_desired = IK_analytics(euler_to_transformation_matrix(x, y, z, 0, 0, 0), theta=theta, type="V")[0]

    else:
        # shink radius
        distar = 50
        x = x - 30
        z = z - 20
        x_dis = distar / np.sqrt(x**2 + y**2) * x
        y_dis = distar / np.sqrt(x**2 + y**2) * y

        wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
        # wp2 = IK_analytics(euler_to_transformation_matrix(x - x_dis, y - y_dis , z, 0, 0, 0), theta=theta, type="H")[0]
        wp2 = np.array([q_desired[0], q_desired[1] - 1, q_desired[2] - 1, q_desired[3] - 1, 0])

    if fast:
        waypoints = [wp1, q_desired]
    else:
        if type == "place":
            waypoints = [wp1, wp2, q_desired, wp2]
        else:
            waypoints = [wp1, wp2, q_desired]
    return waypoints    
    # """
    # Create a waypoint for the robot to grab an object.
    
    # Args:
    # target_xyz: Translation coordinates
    # theta (float): Rotation angle in radians
    
    # Returns:
    # list of np.ndarray: List of joint angles for the robot to reach the waypoint
    # """
    # x, y, z = target_xyz

    # # compensation
    # if type == "grab":
    #     z = z - 15
    # elif type == "place":
    #     z = z + 15

    # z = max(z, 10)
    # print(f"Click to reach at {x,y,z}")
    # T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
    # # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
    # q_desired, reach_type = IK_analytics(T_desired, theta=theta)
    # if type == "place" and z < 220:
    #     reach_type = "V"
    # elif type == "place":
    #     reach_type  = "H"
    # else:
    #     q_desired, reach_type = IK_analytics(T_desired, theta=theta)

    # if reach_type == "V":
    #     # gravity compensation
    #     r = np.sqrt(x**2 + y**2)
    #     cooef= np.array([-1.450000e-06,  5.735600e-04, -7.048614e-02,  6.217500e-02])
    #     compensation = cooef[0]*r**3 + cooef[1]*r**2 + cooef[2]*r + cooef[3]
    #     z = z - compensation

    #     wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
    #     wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+30, 0, 0, 0), theta=theta, type="V")[0]
    #     q_desired = IK_analytics(euler_to_transformation_matrix(x, y, z, 0, 0, 0), theta=theta, type="V")[0]

    # else:
    #     # shink radius
    #     distar = 50
    #     x_dis = distar / np.sqrt(x**2 + y**2) * x
    #     y_dis = distar / np.sqrt(x**2 + y**2) * y

    #     z = z - 15 - 20
    #     q_desired = IK_analytics(euler_to_transformation_matrix(x, y, z, 0, 0, 0), theta=theta, type="H")[0]
    #     wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
    #     # wp2 = IK_analytics(euler_to_transformation_matrix(x - x_dis, y - y_dis , z, 0, 0, 0), theta=theta, type="H")[0]
    #     wp2 = np.array([q_desired[0], q_desired[1] - 1, q_desired[2] - 1, q_desired[3] - 1, 0])

    # if fast:
    #     waypoints = [wp1, q_desired]
    # else:
    #     if type == "place":
    #         waypoints = [wp1, wp2, q_desired, wp2]
    #     else:
    #         waypoints = [wp1, wp2, q_desired]
    # return waypoints

def build_waypoint(target_xyz, theta=0, type=None, fast = False):
    """
    Create a waypoint for the robot to grab an object.
    
    Args:
    target_xyz: Translation coordinates
    theta (float): Rotation angle in radians
    
    Returns:
    list of np.ndarray: List of joint angles for the robot to reach the waypoint
    """
    x, y, z = target_xyz

    # Translation and rotation
    z = max(z,0) / 9 * 10 # error from realsense

    # compensation
    if type == "grab":
        z = z - 15
    # elif type == "place":
    #     z = z + 30

    # do gravity compensation, linearly depend on the distance to the center
    # radius = np.sqrt(x**2 + y**2)
    # compensation_z = - 0.5668 * radius + 123.79
    # z = z - compensation_z
    # safe consideration
    z = max(z, 10)
    print(f"Click to reach at {x,y,z}")
    T_desired = euler_to_transformation_matrix(x,y,z,0,0,0)
    # q_desired = IK_with_fixed_base_yaw(x, y, z+40, 0, 0, self.rxarm.S_list, self.rxarm.M_matrix, num_starts=1000)
    q_desired, reach_type = IK_analytics(T_desired, theta=theta)
    if reach_type == "V":
        # gravity compensation
        r = np.sqrt(x**2 + y**2)
        cooef= np.array([-1.450000e-06,  5.735600e-04, -7.048614e-02,  6.217500e-02])
        compensation = cooef[0]*r**3 + cooef[1]*r**2 + cooef[2]*r + cooef[3]
        z = z - compensation


    wp1 = np.array([q_desired[0], 0, -np.pi/2, 0, 0])
    wp2 = IK_analytics(euler_to_transformation_matrix(x, y, z+50, 0, 0, 0), theta=theta)[0]

    if fast:
        waypoints = [wp1, q_desired]
    else: 
        waypoints = [wp1, wp2, q_desired]
    return waypoints







############################################################################################################
############################################################################################################
# detection
############################################################################################################
############################################################################################################



font = cv2.FONT_HERSHEY_SIMPLEX
colors = list((
    {'id': 'red', 'color': (19, 15, 110)},
    {'id': 'red', 'color': (46, 23, 183)},
    {'id': 'red', 'color': (62, 41, 134)},
    {'id': 'red', 'color': (64, 54, 187)},
    {'id': 'red', 'color': (52, 40, 86)},
    {'id': 'orange', 'color': (25, 80, 170)},
    {'id': 'orange', 'color': (63, 90, 209)},
    {'id': 'orange', 'color': (58, 75, 173)},
    {'id': 'orange', 'color': (23, 68, 150)},
    {'id': 'yellow', 'color': (30, 150, 200)},
    {'id': 'green', 'color': (89, 124, 48)}, 
    {'id': 'green', 'color': (118, 131, 83)},
    {'id': 'green', 'color': (74, 85, 25)},
    {'id': 'green', 'color': (48, 74, 33)},
    {'id': 'blue', 'color': (116, 83, 38)},
    {'id': 'blue', 'color': (110, 64, 2)},
    {'id': 'violet', 'color': (118, 70, 52)}, 
    {'id': 'violet', 'color': (100, 40, 80)}, 
    {'id': 'violet', 'color': (47, 35, 40)}, 
    {'id': 'violet', 'color': (72, 57, 65)}, 
))
def retrieve_area_color(data, contour, labels):
    mask = np.zeros(data.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean = cv2.mean(data, mask=mask)[:3]
    min_dist = (np.inf, None)
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1] 

def depth_image_to_world(depth_img, intrinsic_matrix, homogenous_transform):
    """
    Convert a depth image to an array of 3D world coordinates.
    
    Parameters:
      depth_img:         (H, W) numpy array of depth values.
      intrinsic_matrix:  (3, 3) camera intrinsics matrix.
      homogenous_transform: (4, 4) homogeneous transformation matrix 
                            (typically the camera-to-world transform).
    
    Returns:
      world_points: (H, W, 3) numpy array where each (i,j) element is the
                    3D world coordinate corresponding to that pixel.
    """
    H, W = depth_img.shape

    # Step 1: Create a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u: x-coordinate, v: y-coordinate
    ones = np.ones_like(u)
    
    # Stack to create homogeneous pixel coordinates (3, H*W)
    pixel_coords = np.stack([u, v, ones], axis=0).reshape(3, -1)
    
    # Step 2: Back-project pixels to camera coordinates
    # Compute inverse of the intrinsic matrix
    K_inv = np.linalg.inv(intrinsic_matrix)
    
    # Multiply pixel coordinates by K_inv and scale by the depth
    # Flatten the depth image to match the pixel grid
    depths = depth_img.reshape(-1)  # (H*W,)
    camera_points = K_inv @ pixel_coords  # (3, H*W)
    camera_points *= depths  # scale each column by its depth

    # Step 3: Convert camera coordinates to homogeneous coordinates
    camera_points_hom = np.vstack([camera_points, np.ones((1, camera_points.shape[1]))])
    
    # Step 4: Transform to world coordinates
    # If your homogenous_transform transforms from world to camera,
    # you need its inverse to go from camera to world.
    world_points_hom = np.linalg.inv(homogenous_transform) @ camera_points_hom

    # Step 5: Normalize to get Cartesian coordinates (divide by the last row)
    world_points = world_points_hom[:3] / world_points_hom[3]
    
    # Reshape back to the image shape, so that each pixel has a 3D coordinate
    world_points_img = world_points.T.reshape(H, W, 3)
    
    return world_points_img, world_points_img[:, :, 2] # return only the z-axis


def detection(rgb_image, cnt_image, depth_data, homogenous_matrix, intrinsic_matrix, homography_matrix,
              lower = 10, upper = 150):
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cnt_image = cv2.cvtColor(cnt_image, cv2.COLOR_BGR2RGB)

    # align depth image with rgb image
    src_points = np.array([
    [525, 564],
    [577, 397],
    [872, 389],
    [792, 310],
    ], dtype=np.float32)

    # Define the destination points (transformed coordinates)
    dst_points = np.array([
        [516, 567],
        [569, 397],
        [867, 390],
        [787, 307],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points)
    depth_data = cv2.warpPerspective(depth_data, H, (rgb_image.shape[1], rgb_image.shape[0])) 

    # to world coordinates
    world_coord, depth_data = depth_image_to_world(depth_data, intrinsic_matrix, homogenous_matrix)

    # homography
    depth_data = cv2.warpPerspective(depth_data, homography_matrix, (cnt_image.shape[1], cnt_image.shape[0]))
    world_coord = cv2.warpPerspective(world_coord, homography_matrix, (cnt_image.shape[1], cnt_image.shape[0]))
    # rgb_image = cv2.warpPerspective(rgb_image, homography_matrix, (cnt_image.shape[1], cnt_image.shape[0])) # already transformed
    # cnt_image = cv2.warpPerspective(cnt_image, homography_matrix, (cnt_image.shape[1], cnt_image.shape[0])) # already transformed

    # mask and threshold
    mask = np.zeros_like(depth_data, dtype=np.uint8)
    cv2.rectangle(mask, (110,0),(1170,710), 255, cv2.FILLED) # outer boundary
    cv2.rectangle(mask, (560,370),(730,720), 0, cv2.FILLED) # robo arm space
    cv2.rectangle(cnt_image, (110,0),(1170,710), (255, 0, 0), 2) # outer boundary
    cv2.rectangle(cnt_image, (560,370),(730,720), (255, 0, 0), 2) # robo arm space
    thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
    # depending on your version of OpenCV, the following line could be:
    # _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []

    # draw
    # cv2.drawContours(cnt_image, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        # Reshape contour to a 2D array of points (n_points x 2)
        pts = contour.reshape(-1, 2)
        # Get the depth for each contour point.
        # (Make sure the points are valid indices in depth_data)
        # Using (y, x) indexing for the depth image.
        depth_values = depth_data[pts[:, 1], pts[:, 0]]
        # meandian of top10:
        max_depth = np.median(np.sort(depth_values)[-20:])

        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue  # Skip if the contour has zero area

        # Create a mask for the circular area with a radius of 10 pixels
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.circle(mask, (cX, cY), 10, 255, -1)

        # Extract the depth values within this circular area
        depth_values1 = depth_data[mask == 255]


        # Filter points that have depth in the range [max_depth-5, max_depth]
        in_range = (depth_values >= (max_depth - 5)) & (depth_values <= max_depth)
        filtered_pts = pts[in_range]
        # If not enough points remain, skip this contour
        if len(filtered_pts) < 3:
            continue
        # Optional: If you require a closed contour, you might want to ensure the
        # first and last points are the same. For many purposes, this is not necessary.
        if not np.array_equal(filtered_pts[0], filtered_pts[-1]):
            filtered_pts = np.vstack([filtered_pts, filtered_pts[0]])
        # Reshape to the format expected by OpenCV (n_points x 1 x 2)
        contour = filtered_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(cnt_image, [contour], -1, (0,255,255), thickness=1)

        center, edge, theta = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)

        if area < 400:
            continue
        min_rect_area = edge[0]*edge[1]

        # check if square block
        if abs(edge[0]-edge[1]) / max(edge[0], edge[1]) > 0.2:
            continue
        if area / min_rect_area < 0.8:
            continue
        
        # check mode and mean depth in area are similar
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        depth_values = depth_data[mask == 255]
        if np.median(depth_values) / max_depth < 0.95:
            continue
        

        # Calculate the median of these depth values
        max_depth = np.median(np.sort(depth_values1)[-20:])
        # world coordinates
        world_xyz = world_coord[int(center[1]), int(center[0]), :]
        world_xyz[2] = max_depth #roboust to noise

        color = retrieve_area_color(rgb_image, contour, colors)
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(cnt_image, color, (cx-30, cy+40), font, 0.8, (0,0,0), thickness=2)
        # cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
        # xyz
        cv2.putText(cnt_image, f"{world_xyz[0]:.1f}, {world_xyz[1]:.1f}, {world_xyz[2]:.1f}", (cx-30, cy-20), font, 0.5, (255,255,255), thickness=2)

        if area < 1200:
            cv2.putText(cnt_image, "small", (cx-30, cy+20), font, 0.8, (255,0,0), thickness=2)
            res.append((color, int(theta), world_xyz, 'small'))
        else:
            cv2.putText(cnt_image, "large", (cx-30, cy+20), font, 0.8, (255,0,0), thickness=2)
            res.append((color, int(theta), world_xyz, 'large'))

        # print(color, int(theta), cx, cy)
        


    cnt_image = cv2.cvtColor(cnt_image, cv2.COLOR_RGB2BGR)
    return cnt_image, res
    



   