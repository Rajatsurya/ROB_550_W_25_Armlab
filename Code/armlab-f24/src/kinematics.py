"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
# from utils import DTYPE, Quaternion, rot_to_rpy
from math import sin, cos


M_matrix= np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 424.15],
            [0.0, 0.0, 1.0, 303.91],
            [0.0, 0.0, 0.0, 1.0]
        ])

# Screw vectors
S_list = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, -103.91, 0.0],
    [-1.0, 0.0, 0.0, 0.0, -303.91, 50],
    [-1.0, 0.0, 0.0, 0.0, -303.91, 250],
    [0.0, 1.0, 0.0, -303.91, 0.0, 0.0]
]) #t7values

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """

    T = np.eye(4)
    s_lst = s_lst.T
    for i in range(len(joint_angles)):
        screw_axis = s_lst[:, i]
        w=screw_axis[:3]
        v=screw_axis[3:]
        S_matrix = to_s_matrix(w,v)
        T = np.dot(T, expm(S_matrix * joint_angles[i]))
    T= np.dot(T, m_mat)
    return T

def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """

    # Create a 4x4 matrix for se(3)
    skew_matrix = np.array([
        [0, -w[2], w[1], v[0]], 
        [w[2], 0, -w[0], v[1]], 
        [-w[1], w[0], 0, v[2]], 
        [0, 0, 0, 0]
    ])
    return skew_matrix


# def get_euler_angles_from_T(T):
#     """!
#     @brief      Gets the euler angles from a transformation matrix.

#                 TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
#                 If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

#     @param      T     transformation matrix

#     @return     The euler angles from T.
#     """
#     R = T[:3, :3]  # Extract the rotation matrix

#     # Compute theta (pitch)
#     if -1 <= -R[1, 2] <= 1:  # Normal case
#         theta = np.arcsin(-R[1, 2])
#     else:  # Handle numerical precision issues
#         theta = np.pi / 2 if -R[1, 2] > 0 else -np.pi / 2

#     # Check for gimbal lock
#     if np.isclose(np.abs(theta), np.pi / 2):  # Gimbal lock condition
#         phi = 0  # Arbitrary value for roll
#         psi = np.arctan2(R[0, 1], R[0, 0])  # Simplified yaw calculation
#     else:
#         # Normal case
#         psi = np.arctan2(R[0, 1], R[0, 0])  # Yaw (z-axis rotation)
#         phi = np.arctan2(R[2, 2], R[0, 2])  # Roll (y-axis rotation)

#     return [phi - np.pi/2, theta, psi]
#    # roll pitch yaw

def get_euler_angles_from_T(T):
    """
    Extract Z-Y-Z Euler angles from 4x4 transformation matrix.
    Returns: alpha (first Z), beta (Y), gamma (second Z) in radians
    """
    R = T[:3, :3]
    
    beta = np.arctan2(np.sqrt(1-R[2, 2]**2), R[2, 2])
    alpha = np.arctan2(R[1, 2], R[0, 2])
    gamma = np.arctan2(R[2, 1], -R[2, 0])
        
    return alpha, beta, gamma



def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """

    #so there is a point which is represented in the coordinates of the grippers end effector frame and now I have to represent its pose in the world frame using the tasformation matrix I have obtained 
    #and then return the point in world cordinates xyz and the rotation angles to obtain that gripper position that is roll pitch yaw have I intrepreted this function properly?
    # xyz = T[:3, 3]
    # rpy = get_euler_angles_from_T_r_p_y(T)
    # pose = np.hstack((xyz, rpy))
    # return pose

    xyz = T[:3, 3]
    [yaw,pitch,roll] = get_euler_angles_from_T(T)
    pose = np.hstack((xyz, [yaw,pitch,roll] ))
    return pose



def adjoint(T):
    """
    Compute the 6x6 adjoint representation of a transformation matrix T.
    
    The adjoint maps a twist expressed in one frame to a twist expressed in another.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    p_skew = np.array([
        [   0,   -p[2],  p[1]],
        [ p[2],      0, -p[0]],
        [-p[1],   p[0],     0]
    ])
    adj_T = np.zeros((6, 6))
    adj_T[:3, :3] = R
    adj_T[3:, :3] = p_skew @ R
    adj_T[3:, 3:] = R
    return adj_T

def compute_jacobian(joint_angles, s_lst):
    """
    Compute the spatial Jacobian for the current configuration.
    
    Parameters:
      joint_angles: (n,) current joint angles.
      s_lst: n x 6 matrix of screw axes.
      
    Returns:
      J: 6 x n Jacobian matrix.
    """
    n = len(joint_angles)
    J = np.zeros((6, n))
    T = np.eye(4)
    for i in range(n):
        # For the i-th joint, map the screw axis into the current frame:
        screw = s_lst[i, :]   # (6,)
        J[:, i] = adjoint(T) @ screw
        # Update T by “moving” along the i-th joint’s motion:
        S_matrix = to_s_matrix(screw[:3], screw[3:])
        T = T @ expm(S_matrix * joint_angles[i])
    return J

def se3_to_vec(se3mat):
    """
    Convert a 4x4 se(3) matrix into a 6-vector (the vee operator).
    
    The se(3) matrix is assumed to have the form:
      [ 0    -w3   w2   v1 ]
      [ w3    0   -w1   v2 ]
      [-w2   w1    0    v3 ]
      [ 0     0    0     0 ]
    and returns the 6-vector: [w1, w2, w3, v1, v2, v3].
    """
    # Ensure we work with real numbers (logm may return small imaginary parts)
    se3mat = np.real(se3mat)
    w = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    v = np.array([se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]])
    return np.concatenate((w, v))


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass

def IK_pox_numeric(T_desired, m_mat, s_lst, initial_guess=None, tol=5e-3, max_iterations=5000, alpha=0.2):
    """
    Compute the inverse kinematics for the given end-effector transformation using
    an iterative, Jacobian-based approach.
    
    Parameters:
      T_desired: 4x4 homogeneous transformation matrix representing the desired pose.
      m_mat: 4x4 home configuration matrix.
      s_lst: n x 6 matrix of screw axes.
      initial_guess: (n,) vector of initial joint angles (if None, zeros are used).
      tol: Tolerance for the norm of the error twist.
      max_iterations: Maximum number of iterations.
      alpha: Step size (gain) for the update.
      
    Returns:
      joint_angles: (n,) vector of joint angles that (approximately) achieve T_desired.
    """
    from scipy.linalg import logm
    n = s_lst.shape[0]  # number of joints
    if initial_guess is None:
        joint_angles = np.zeros(n)
    else:
        joint_angles = initial_guess.copy()

    for i in range(max_iterations):
        # 1. Compute the current end-effector transformation.
        T_current = FK_pox(joint_angles, m_mat, s_lst)
        
        # 2. Compute the error transformation:
        # T_error = np.linalg.inv(T_current) @ T_desired
        T_error = T_desired @ np.linalg.inv(T_current)
        # 3. Compute the error twist (as a 6-vector) via the matrix logarithm.
        error_se3 = logm(T_error)
        error_twist = se3_to_vec(error_se3)
        
        # Check if the error is small enough:
        error_norm = np.linalg.norm(error_twist)
        if error_norm < tol:
            print(f"Converged in {i} iterations with error norm {error_norm:.6f}.")
            return joint_angles
        
        # 4. Compute the spatial Jacobian at the current joint angles.
        J = compute_jacobian(joint_angles, s_lst)  # shape: 6 x n
        
        # 5. Compute a change in joint angles (delta_theta) using the pseudoinverse.
        delta_theta = alpha * np.linalg.pinv(J) @ error_twist
        
        # 6. Update the joint angles.
        joint_angles = joint_angles + delta_theta
        
    print("Warning: IK did not converge within the maximum number of iterations.")
    return joint_angles


def euler_to_transformation_matrix(x, y, z, yaw, pitch, roll):
    """
    Build a 4x4 homogeneous transformation matrix from the position (x,y,z)
    and Euler angles (yaw, pitch, roll) using the sequence:
    
       R = R_z(yaw) * R_x(pitch) * R_y(roll)
    
    This is chosen for the coordinate system:
      - x: right
      - y: front
      - z: up
      
    The yaw is a rotation about the z axis.
    """
    # Rotation about z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,           0,            1]
    ])
    # Rotation about x (pitch)
    Rx = np.array([
        [1,          0,           0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    # Rotation about y (roll)
    Ry = np.array([
        [ np.cos(roll), 0, np.sin(roll)],
        [0,             1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    R = Rz @ Rx @ Ry
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

from scipy.optimize import minimize
def IK_with_fixed_base_yaw(target_x, target_y, target_z, target_pitch, target_roll, S_list, M_matrix, num_starts=500000):
    """
    For a robot whose first (base) joint is the only one that can affect yaw,
    compute the required base yaw from the target x,y position. Then build a
    desired transformation with that yaw, and solve for the remaining joints so
    that the end-effector achieves the target translation and pitch.
    
    Parameters:
      target_x, target_y, target_z : desired translation (in mm or units consistent with M_matrix)
      target_pitch, target_roll     : desired pitch and roll (in radians)
      num_starts                    : number of random initial guesses
      
    Returns:
      q_full       : Full joint-angle vector (with the base joint fixed).
      best_obj     : Best achieved objective value.
      base_yaw     : The computed (and fixed) base yaw.
      T_desired_new: The desired transformation (with base yaw applied).
    """
    n = S_list.shape[0]  # Total number of joints (here, 5)
    
    # Compute the required base yaw from target x and y.
    base_yaw = np.arctan2(-target_x, target_y)
    
    # Build the desired transformation with the computed base yaw.
    # (Here, we use the given target translation and the desired pitch/roll.)
    T_desired_new = euler_to_transformation_matrix(target_x, target_y, target_z,
                                                   base_yaw, target_pitch, target_roll)
    desired_pose = get_pose_from_T(T_desired_new)
    
    # We will optimize only the remaining (n-1) joints.
    def objective(q_free):
        # Construct the full joint vector: first joint is fixed to base_yaw.
        q_full = np.hstack(([base_yaw], q_free))
        T_current = FK_pox(q_full, M_matrix, S_list)
        current_pose = get_pose_from_T(T_current)
        # We care about translation (x, y, z) and pitch (index 4) only.
        error = np.array([
            current_pose[0] - desired_pose[0],
            current_pose[1] - desired_pose[1],
            current_pose[2] - desired_pose[2],
            current_pose[4] - desired_pose[4]
        ])
        # Use heavy weighting on translation and pitch.
        weights = np.array([150.0, 150.0, 150.0, 100.0])
        return np.sum((weights * error) ** 2)
    
    # Optimize over the remaining joints with loose bounds.
    bounds = [(-np.pi/2, np.pi/2)] * (n - 1)
    
    best_sol = None
    best_obj = np.inf
    for i in range(num_starts):
        q0 = np.random.uniform(-np.pi, np.pi, size=n - 1)
        res = minimize(objective, q0, method='SLSQP', bounds=bounds,
                       options={'ftol': 8e-9, 'maxiter': 2000})
        q_full = np.hstack(([base_yaw], res.x.copy()))
        z = get_pose_from_T(FK_pox(q_full, M_matrix, S_list))[2]
        if np.isfinite(res.fun) and res.fun < best_obj and z > 5:
            best_obj = res.fun
            best_sol = res.x.copy()
    
    if best_sol is None:
        return None, best_obj, base_yaw, T_desired_new
    else:
        q_full = np.hstack(([base_yaw], best_sol))
        return q_full# , best_obj, base_yaw, T_desired_new
    

from scipy.optimize import fsolve

# def IK_analytics(transformation_matrix):
#     def helper(x,y,z):
#         def equations(p):
#             theta_1, theta_3 = p

#             eq1 = 200 * np.sin(theta_1) + 50 * np.cos(theta_1) + 200 * np.sin(theta_3) - np.sqrt(x**2 + y**2)
#             eq2 = 200 * np.cos(theta_1) + 50 * np.sin(theta_1) - 200 * np.cos(theta_3) - 70.24 - z

#             return [eq1, eq2]

#         initial_guess = [0, 0] 

#         solution = fsolve(equations, initial_guess)

#         theta_1, theta_3 = solution
#         theta_2 = np.pi/2 - theta_1 - theta_3
#         return theta_1, theta_2, theta_3
    
#     x, y, z = transformation_matrix[:3, 3]
#     theta_0 = np.arctan2(-x, y)
#     the1, the2, the3 = helper(x, y, z)
    
#     return np.array([theta_0, the1, the2, the3, theta_0])

def IK_analytics(transformation_matrix, theta=0, type = None):
    x, y ,z = transformation_matrix[:3, 3]
    def equations_v(p):
        theta_1, theta_3 = p

        eq1 = 200 * np.sin(theta_1) + 50 * np.cos(theta_1) + 200 * np.sin(theta_3) - np.sqrt(x**2 + y**2)
        eq2 = 200 * np.cos(theta_1) - 50 * np.sin(theta_1) - 200 * np.cos(theta_3) - 70.24 - z

        return [eq1, eq2]


    def equations_h(p):
        theta_1, theta_3 = p

        eq1 = 200 * np.sin(theta_1) + 50 * np.cos(theta_1) + 200 * np.cos(theta_3) + 174.15 - np.sqrt(x**2 + y**2)
        eq2 = 200 * np.cos(theta_1) - 50 * np.sin(theta_1) + 200 * np.sin(theta_3) + 103.91 - z

        return [eq1, eq2]

    initial_guess = [0, 0]

    sol_v = fsolve(equations_v, initial_guess)
    error_v = equations_v(sol_v)

    sol_h = fsolve(equations_h, initial_guess)
    error_h = equations_h(sol_h)

    flag = "Error"

    radius = np.sqrt(x**2 + y**2)


    # test for now

    if type == "V" or (radius < 350 and z < 100):
        print("Reach position vertically")
        theta_0 = np.arctan2(-x, y)
        theta_1, theta_3 = sol_v
        theta_2 = np.pi/2 - theta_1 - theta_3
        theta_4 = theta_0
        error = np.linalg.norm(error_v)
        theta_4 = theta_4 + theta / 180 * np.pi

        if error < 1e-3:
            flag = "V"
    else:
        print("Reach position horizontally")
        theta_0 = np.arctan2(-x, y)
        theta_1, theta_3 = sol_h
        theta_2 = - theta_3 - theta_1
        theta_4 = 0
        error = np.linalg.norm(error_h)
        if error < 1e-3:
            flag = "H"
    
    
    # validate
    true_ee_transform = FK_pox([theta_0, theta_1, theta_2, theta_3, theta_4], M_matrix, S_list)
    true_ee_xyz = true_ee_transform[:3, 3]
    error = np.linalg.norm(true_ee_xyz - transformation_matrix[:3, 3])
    print("Error: ", error)
    if error > 1e-3:
        print("Using {flag} solution but the error is too large")
        breakpoint()

    return np.array([theta_0, theta_1, theta_2, theta_3, theta_4]), flag
    # print(theta_1, theta_2, theta_3)

    # degree = np.array([theta_1, theta_2, theta_3]) * 180 / np.pi
    # print(degree)


