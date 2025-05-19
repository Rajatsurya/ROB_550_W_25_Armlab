
#!/usr/bin/python
""" Example: 

python3 label_blocks.py -i image_blocks.png -d depth_blocks.png -l 905 -u 973

"""
import argparse
import sys
import cv2
import numpy as np


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
    value = None
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
            value = mean
    return min_dist[1], mean



def find_thres(depth):
    return np.min(depth), np.max(depth)
    
args = {
    "image": "launch/output.png",
    "depth": "launch/output_depth.png",
    "lower": 800, #900,
    "upper": 1042  #973
}

lower = int(args["lower"])
upper = int(args["upper"])
rgb_image = cv2.imread(args["image"])
cnt_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# show original

# Define the source points (original coordinates)
src_points = np.array([
    [344, 247], 
    [930, 168], 
    [967, 436], 
    [526, 463]
], dtype=np.float32)

# Define the destination points (transformed coordinates)
dst_points = np.array([
    [330, 242], 
    [926, 161], 
    [963, 434], 
    [518, 466]
], dtype=np.float32)
# Compute the homography matrix H (source → destination)
H, _ = cv2.findHomography(src_points, dst_points)
depth_data = cv2.warpPerspective(depth_data, H, (rgb_image.shape[1], rgb_image.shape[0]))


# import numpy as np
homogenous= np.array([[ 9.99928300e-01, -8.06682230e-03, -8.85029789e-03,  2.19621067e+01], 
 [-6.81044773e-03, -9.90985386e-01,  1.33797050e-01,  1.48745499e+02], 
 [-9.84983201e-03, -1.33727174e-01, -9.90969241e-01,  1.01971588e+03], 
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

intrinsic_matrix = np.array([908.61087, 0.0, 642.62399, 
                                            0.0, 908.59074, 358.7872, 
                                            0.0, 0.0, 1.0], dtype=np.float64).reshape((3, 3)) #this is to test it at the moment


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
    
    return world_points_img[:, :, 2]

# Example usage:
# world_coords = depth_image_to_world(depth_img, intrinsic_matrix, homogenous_transform)

depth_data = depth_image_to_world(depth_data, intrinsic_matrix, homogenous)

homography = np.array(
[[ 1.20086728e+00, -7.28574458e-02, -1.60277031e+02],
 [ 1.93202670e-02,  1.15009887e+00, -8.35685461e+01],
 [ 3.02519543e-05, -1.03022910e-04,  1.00000000e+00]]
)

transformed_image = cv2.warpPerspective(rgb_image, homography, (depth_data.shape[1], depth_data.shape[0]))
transformed_depth = cv2.warpPerspective(depth_data, homography, (depth_data.shape[1], depth_data.shape[0]))
transformed_cnt = cv2.warpPerspective(cnt_image, homography, (depth_data.shape[1], depth_data.shape[0]))
#  cut off the depth by min/max
# transformed_depth = np.clip(transformed_depth, lower, np.max(transformed_depth))

# normalize the depth for visualization
# depth_viz = cv2.normalize(transformed_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

rgb_image, depth_data, cnt_image = transformed_image, transformed_depth, transformed_cnt


cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
# adjust windows size
cv2.resizeWindow("Image window", 1000, 800)
#cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
"""mask out arm & outside board"""
mask = np.zeros_like(depth_data, dtype=np.uint8)
cv2.rectangle(mask, (110,0),(1160,710), 255, cv2.FILLED) # outer boundary
cv2.rectangle(mask, (560,370),(730,720), 0, cv2.FILLED) # robo arm space
cv2.rectangle(cnt_image, (110,0),(1160,710), (255, 0, 0), 2) # outer boundary
cv2.rectangle(cnt_image, (560,370),(730,720), (255, 0, 0), 2) # robo arm space
thresh = cv2.bitwise_and(cv2.inRange(depth_data, 10, 200), mask)
# depending on your version of OpenCV, the following line could be:
# _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)


# def show_rgb(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:  # When mouse moves
#         rgb = rgb_image[y, x]  # Get RGB values
#         print(f"RGB at ({x}, {y}): {rgb}")

# # Create window and set mouse callback``
# cv2.namedWindow('RGB Image')
# cv2.setMouseCallback('RGB Image', show_rgb)

# while True:
#     cv2.imshow('RGB Image', rgb_image)
#     if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
#         break



for contour in contours:
    # Reshape contour to a 2D array of points (n_points x 2)
    pts = contour.reshape(-1, 2)
    # Get the depth for each contour point.
    # (Make sure the points are valid indices in depth_data)
    # Using (y, x) indexing for the depth image.
    depth_values = depth_data[pts[:, 1], pts[:, 0]]
    # Compute the maximum depth among these points.
    max_depth = np.max(depth_values)
    # meandian of top10:
    max_depth = np.median(np.sort(depth_values)[-20:])
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

    color, rgb = retrieve_area_color(rgb_image, contour, colors)
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
    max_depth = depth_values.max()
    if np.median(depth_values) / np.max(depth_values) < 0.8:
        continue

    
    M = cv2.moments(contour)
    if M['m00'] == 0:
        continue
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=1)
    cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=1)
    # cv2.putText(cnt_image, str(int(area)), (cx, cy+20), font, 0.5, (255, 0, 0), thickness=1)
    # cv2.putText(cnt_image, str(int(min_rect_area)), (cx, cy+30), font, 0.5, (0, 255, 0), thickness=1)
    if area < 1200:
        cv2.putText(cnt_image, "small", (cx-30, cy+20), font, 0.5, (255,0,0), thickness=1)
    else:
        cv2.putText(cnt_image, "large", (cx-30, cy+20), font, 0.5, (255,0,0), thickness=1)

    print(color, int(theta), cx, cy, rgb)


# cv2.imshow("Threshold window", thresh)
cv2.imshow("Image window", cnt_image)
while True:
  k = cv2.waitKey(0)
  if k == 27:  # quit with ESC
    break    
cv2.destroyAllWindows()