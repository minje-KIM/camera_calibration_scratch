import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
import os
import util as util
from cv2 import aruco

def calibrate():
    
    pass
    


def solve_extrinsic(intrinsic, homographies):
    """
    Find out the extrinsics of each image by intrinsic and homographyies
    
    Return:
        extrinsics: the extrensics matrix of each image
    """

    extrinsics = []
    inv_intrinsic = np.linalg.inv(intrinsic)
    for i in range(len(homographies)): # Iteratively find out the extrinsics of each image
        #h1, h2, h3 are columns in H
        h1 = homographies[i][:,0]
        h2 = homographies[i][:,1]
        h3 = homographies[i][:,2]
        
        lamda1 = 1/np.linalg.norm(np.dot(inv_intrinsic, h1))
        lamda2 = 1/np.linalg.norm(np.dot(inv_intrinsic, h2))
        lamda3 = (lamda1 + lamda2) / 2
        
        r1 = lamda1 * np.dot(inv_intrinsic, h1)
        r2 = lamda2 * np.dot(inv_intrinsic, h2)
        r3 = np.cross(r1, r2)
        t = lamda3 * np.dot(inv_intrinsic, h3)
        
        RT = np.array([r1, r2, r3, t]).transpose()
        extrinsics.append(RT)
        
    extrinsics = np.array(extrinsics)    
    return extrinsics  

def solve_homography(P, m):

    A = []  

    for r in range(len(P)):
        A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0][0], P[r,1]*m[r,0][0], m[r,0][0]])
        A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,0][1], P[r,1]*m[r,0][1], m[r,0][1]])
    
    u, s, vt = np.linalg.svd(A)
    H = np.reshape(vt[-1], (3,3))
    H = (1/H.item(-1)) * H
    
    return H

def solve_intrinsic(homographies):
    """
    Use homographies to find out the intrinsic matrix K  

    Return: 
        K: intrisic matrix
        B: B = K^(-T) * K^(-1) 
    """

    v = []
    for i in range(len(homographies)):
        h = homographies[i]
        v.append(v_pq(0, 1, h))
        v.append(np.subtract(v_pq(0, 0, h), v_pq(1, 1, h)))
    
    v = np.array(v)
    u, s, vh = np.linalg.svd(v)
    b = vh[-1]

    # Make sure that B is positive definite
    if(b[0] < 0 or b[2] < 0 or b[5] < 0):
        b = -b
   
    B = [[b[0], b[1], b[3]],
         [b[1], b[2], b[4]],
         [b[3], b[4], b[5]]]
    B = np.array(B)
  
    # Solve K by cholesky factoriztion where B=K^(-T)K^(-1), L=K^(-T)
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L).transpose() * L[2,2]
    
    return K, B

def v_pq(p, q, H):
    v = np.array([
            H[0, p]*H[0, q],
            H[0, p]*H[1, q] + H[1, p]*H[0, q], 
            H[1, p]*H[1, q],
            H[2, p]*H[0, q] + H[0, p]*H[2, q],
            H[2, p]*H[1, q] + H[1, p]*H[2, q],
            H[2, p]*H[2, q]
        ])
    return v

def detectchessboard(num_corner_x, num_corner_y, dir_chess, dir_result):
    
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    dir_board_corner = os.path.join(dir_result, "chessboard_corner")
    if not os.path.exists(dir_board_corner):
        os.makedirs(dir_board_corner)

    # initialize the world coordinates for chessboard
    objp = np.zeros((num_corner_x*num_corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corner_x, 0:num_corner_y].T.reshape(-1,2)  #  (num_corner_x * num_corner_y , 3)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(f'{dir_chess}/*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)

        ret, corners = cv2.findChessboardCorners(gray, (num_corner_x,num_corner_y), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # print(corners.shape)     # (num_corner_x * num_corner_y, 1, 2)

            for i, corner in enumerate(corners):
                center = tuple(map(int, corner[0]))  
                cv2.circle(img, center, radius=16, color=(0, 0, 255), thickness=-1)  
                
                # TODO: Update the text size appropriately for the input image resolution.
                # Currently, when displaying the 3D and 2D coordinates of each corner point as text, 
                # the text size varies depending on the image resolution. 
                # It is necessary to write code to dynamically adjust the text size to ensure 
                # it is visually appropriate for the image size
                
                # Add objpoint (3D coordinate) above the corner
                obj_text = f"{int(objp[i, 0])}, {int(objp[i, 1])}, {int(objp[i, 2])}"
                cv2.putText(img, obj_text, (center[0] - 120, center[1] - 40), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

                # Add imgpoint (2D coordinate) below the corner
                img_text = f"{corner[0, 0]:.1f}, {corner[0, 1]:.1f}"
                cv2.putText(img, img_text, (center[0] - 120, center[1] + 40), cv2.FONT_ITALIC, 1, (255, 0, 255), 3)

        output_path = f"{dir_result}/chessboard_corner/corner_chessboard_{idx}.jpg"
        # cv2.drawChessboardCorners(img, (num_corner_x, num_corner_y), corners, ret)
        cv2.imwrite(output_path, img)  

    return objpoints, imgpoints

def detectarucoboard(dir_aruco, aruco_dict, dir_result, markerLength, markerSeparation, s_vertical, s_horizontal):

    board = aruco.GridBoard_create(s_vertical, s_horizontal, markerLength, markerSeparation, aruco_dict)

    arucoParams = aruco.DetectorParameters_create()

    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    dir_board_corner = os.path.join(dir_result, "arucoboard_corner")
    if not os.path.exists(dir_board_corner):
        os.makedirs(dir_board_corner)

    # Initialize parameters
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob(f'{dir_aruco}/*.JPG')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)

        if ids is not None and len(ids) > 0:
            # Extract object points for the detected markers
            objp = []
            for i, id_ in enumerate(ids):
                # `board.objPoints` provides the object points for each marker
                objp.append(board.objPoints[id_[0]])  # Append 4 corner points for each marker
            
            objp = np.vstack(objp).reshape(-1, 3)  
            objpoints.append(objp)

            # Extract image points for the detected corners
            corners_flatten = np.vstack(corners).reshape(-1, 1, 2)  # Flatten to shape (n, 1, 2)
            imgpoints.append(corners_flatten)

            for i in range(len(objp)):
                # 2D image point
                point2D = tuple(corners_flatten[i][0].astype(int))
                # 3D object point
                point3D = objp[i]

                cv2.circle(img, point2D, radius=5, color=(0, 0, 255), thickness=-1)

                # TODO: Update the text size appropriately for the input image resolution.
                # Currently, when displaying the 3D and 2D coordinates of each corner point as text, 
                # the text size varies depending on the image resolution. 
                # It is necessary to write code to dynamically adjust the text size to ensure 
                # it is visually appropriate for the image size

                cv2.putText(
                    img,
                    f"({point3D[0]:.2f}, {point3D[1]:.2f}, {point3D[2]:.2f})",
                    (point2D[0] , point2D[1] - 11),
                    cv2.FONT_ITALIC,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"({point2D[0]}, {point2D[1]})",
                    (point2D[0], point2D[1] + 11),
                    cv2.FONT_ITALIC,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                
            output_path = f"{dir_result}/arucoboard_corner/corner_aruco_{idx}.jpg"
            cv2.imwrite(output_path, img)

    return objpoints, imgpoints

def detectcharucoboard(dir_charuco, aruco_dict, dir_result, squareLegth, markerLength, s_vertical, s_horizontal):

    board = aruco.CharucoBoard_create(s_vertical, s_horizontal, squareLegth, markerLength, aruco_dict)

    arucoParams = aruco.DetectorParameters_create()

    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    dir_board_corner = os.path.join(dir_result, "charucoboard_corner")
    if not os.path.exists(dir_board_corner):
        os.makedirs(dir_board_corner)

    # Initialize parameters
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob(f'{dir_charuco}/*.JPG')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)

        if ids is not None and len(ids) > 0:
            # Interpolate Charuco corners
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)

            # TODO: Modify the code to handle cases where the entire Charuco board is not visible
            # Currently, if the entire Charuco board is not visible in an image, the 2D-3D point pairs do not match, 
            # so such images are excluded. It is necessary to modify the code to include these images as well.
            if ret > 0 and charuco_corners.shape[0] == board.chessboardCorners.shape[0]:
                # Append object points and image points
                objpoints.append(board.chessboardCorners)
                imgpoints.append(charuco_corners)

                # Draw Charuco board corners on the image
                for i in range(len(charuco_corners)):
                    point2D = tuple(charuco_corners[i][0].astype(int))
                    point3D = board.chessboardCorners[charuco_ids[i][0]]

                    cv2.circle(img, point2D, radius=5, color=(0, 0, 255), thickness=-1)

                    # TODO: Update the text size appropriately for the input image resolution.
                    # Currently, when displaying the 3D and 2D coordinates of each corner point as text, 
                    # the text size varies depending on the image resolution. 
                    # It is necessary to write code to dynamically adjust the text size to ensure 
                    # it is visually appropriate for the image size
                    cv2.putText(
                        img,
                        f"({point3D[0]:.2f}, {point3D[1]:.2f}, {point3D[2]:.2f})",
                        (point2D[0], point2D[1] - 11),
                        cv2.FONT_ITALIC,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        f"({point2D[0]}, {point2D[1]})",
                        (point2D[0], point2D[1] + 11),
                        cv2.FONT_ITALIC,
                        0.5,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        output_path = f"{dir_board_corner}/charucoboard_{idx}.jpg"
        cv2.imwrite(output_path, img)

    return objpoints, imgpoints

def compute_jacobian(K, R, t, objpoints):
    num_points = objpoints.shape[0]
    objpoints_h = np.hstack((objpoints, np.ones((num_points, 1))))
    proj = K @ (R @ objpoints_h.T + t.reshape(-1, 1))
    x = proj[0, :] / proj[2, :]
    y = proj[1, :] / proj[2, :]
    z = proj[2, :]

    J_K = np.zeros((num_points, 2, 4))
    J_K[:, 0, 0] = objpoints_h @ R[0, :].T / z
    J_K[:, 1, 1] = objpoints_h @ R[1, :].T / z
    J_K[:, 0, 2] = 1
    J_K[:, 1, 3] = 1

    J_R = np.zeros((num_points, 2, 3))
    J_R[:, 0, 0] = K[0, 0] * (objpoints_h @ R[2, :].T * x - objpoints_h @ R[0, :].T) / z
    J_R[:, 1, 0] = K[1, 1] * (objpoints_h @ R[2, :].T * y - objpoints_h @ R[1, :].T) / z

    J_t = np.zeros((num_points, 2, 3))
    J_t[:, 0, 0] = K[0, 0] / z
    J_t[:, 1, 1] = K[1, 1] / z
    J_t[:, 0, 2] = -K[0, 0] * x / z
    J_t[:, 1, 2] = -K[1, 1] * y / z

    return J_K, J_R, J_t

def project_points(K, R, t, objpoints):
    """
    Project 3D points onto the image plane using the intrinsic matrix and extrinsic parameters.
    
    Args:
        K: Intrinsic matrix (3x3).
        R: Rotation matrix (3x3).
        t: Translation vector (3x1).
        objpoints: 3D object points (Nx3).
    
    Returns:
        Projected 2D points on the image plane.
    """
    # Construct extrinsic matrix
    extrinsic = np.hstack((R, t.reshape(-1, 1)))  # [R | t]
    
    # Project points
    projected = K @ extrinsic @ np.hstack((objpoints, np.ones((objpoints.shape[0], 1)))).T
    projected /= projected[2, :]  # Normalize by z-coordinate
    
    return projected[:2, :].T  # Return x, y coordinates

def compute_reprojection_error(K, extrinsics, objpoints, imgpoints):
    """
    Compute the reprojection error.
    
    Args:
        K: Intrinsic matrix (3x3).
        extrinsics: List of extrinsic parameters (each a dictionary with R and t).
        objpoints: List of 3D object points for each image (list of arrays).
        imgpoints: List of 2D image points for each image (list of arrays).
    
    Returns:
        Total reprojection error.
    """
    total_error = 0.0
    num_points = 0

    for i in range(len(objpoints)):
        R = extrinsics[i][:, :3]
        t = extrinsics[i][:, 3]
        projected_points = project_points(K, R, t, objpoints[i])
        error = np.linalg.norm(imgpoints[i] - projected_points, axis=1)
        total_error += np.sum(error ** 2)
        num_points += len(objpoints[i])
    
    return total_error / num_points  # Average reprojection error


def update_parameters(K, extrinsics, objpoints, imgpoints, lr=1e-4):
    for i in range(len(extrinsics)):
        R = extrinsics[i][:, :3]
        t = extrinsics[i][:, 3]
        
        objpoints_i = objpoints[i]
        imgpoints_i = imgpoints[i]

        J_K, J_R, J_t = compute_jacobian(K, R, t, objpoints_i)
        projected_points = project_points(K, R, t, objpoints_i)
        error = imgpoints_i - projected_points

        delta_K = -lr * np.sum(J_K.transpose(0, 2, 1) @ error[:, :, None], axis=0).flatten()
        K[0, 0] += delta_K[0]
        K[1, 1] += delta_K[1]
        K[0, 2] += delta_K[2]
        K[1, 2] += delta_K[3]

        delta_R = -lr * np.sum(J_R.transpose(0, 2, 1) @ error[:, :, None], axis=0).flatten()
        delta_t = -lr * np.sum(J_t.transpose(0, 2, 1) @ error[:, :, None], axis=0).flatten()

        R += delta_R.reshape(3, 3)
        t += delta_t

        extrinsics[i][:, :3] = R
        extrinsics[i][:, 3] = t

    return K, extrinsics

def nonlinear_optimization(K_init, extrinsics_init, objpoints, imgpoints, max_iter=10, tol=1e-6):
    K = K_init.copy()
    extrinsics = extrinsics_init.copy()
    prev_error = float('inf')

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")

        K, extrinsics = update_parameters(K, extrinsics, objpoints, imgpoints)

        current_error = compute_reprojection_error(K, extrinsics, objpoints, imgpoints)
        print(f"Reprojection Error: {current_error}")

        if abs(prev_error - current_error) < tol:
            print("Convergence reached.")
            break
        prev_error = current_error

    return K, extrinsics



if __name__ == "__main__":
    
    # Initialize settings
    parser = argparse.ArgumentParser(description='calibration arguments')
    parser.add_argument('--mode', default="single", help='choose camera type: single or stereo')
    parser.add_argument('--board_type', default="charucoboard", help='select board type: chessboard, arucoboard, charucoboard') 
    parser.add_argument('--result_dir', default='./result', help='direction for result') 

    # chessboard settings
    parser.add_argument('--corners_x', default=7, help='input chessboard points in x direction') 
    parser.add_argument('--corners_y', default=7, help='input chessboard points in y direction') 
    parser.add_argument('--chess_data_dir', default='./data/chess_data', help='direction for chessboard data') 

    # arucoboard settings
    parser.add_argument('--aruco_data_dir', default='./data/aruco_data', help='direction for arucoboard data') 
    parser.add_argument('--marker_len', default=3.75, help='marker length in cm') 
    parser.add_argument('--marker_sep_len', default=0.5, help='marker seperation length in cm') 
    parser.add_argument('--squares_vertical', default=4, help='number of squares in vertical') 
    parser.add_argument('--squares_horizontal', default=5, help='number of squares in horizontal') 
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

    # charucoboard settings
    parser.add_argument('--charuco_data_dir', default='./data/charuco_data', help='direction for charucoboard data') 
    parser.add_argument('--charuco_marker_len', default=2, help='marker length in cm') 
    parser.add_argument('--charuco_square_len', default=4, help='marker seperation length in cm') 
    parser.add_argument('--charuco_squares_vertical', default=5, help='number of squares in vertical') 
    parser.add_argument('--charuco_squares_horizontal', default=7, help='number of squares in horizontal') 
    charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    args = parser.parse_args()

    """
        1. Feature point detection (corner detection)
            - Get the 2D-3D corner point correspondences matches
            - Compatible with any board type: chessboard, arucoboard, charucoboard

        Args: 
            - Board setting parameters
            - Image data directory
            
        Returns: 
            - 2D points (list of shape: (N_images))
                - Each element in the list is a numpy array of shape: (num_points, 1, 2)
            - 3D points (list of shape: (N_images))
                - Each element in the list is a numpy array of shape: (num_points, 3)
    """
    if args.board_type == "chessboard":
        objpoints, imgpoints = detectchessboard(args.corners_x, args.corners_y, args.chess_data_dir, args.result_dir)
    elif args.board_type == "arucoboard":
        objpoints, imgpoints = detectarucoboard(args.aruco_data_dir, aruco_dict, args.result_dir, args.marker_len, 
                                                args.marker_sep_len, args.squares_vertical, args.squares_horizontal)
    elif args.board_type == "charucoboard":
        objpoints, imgpoints = detectcharucoboard(args.charuco_data_dir, charuco_dict, args.result_dir, args.charuco_square_len, 
                                                args.charuco_marker_len, args.charuco_squares_vertical, args.charuco_squares_horizontal)
    else:
        print("You set the wrong board type!")


    # 2. Homography estimation (world-image relationship)

    homographies = []
    
    for i in range(len(imgpoints)): # Find out the homography matrix of each iamge
        homographies.append(solve_homography(objpoints[i], imgpoints[i]))
        
    # print(len(homographies))              #  number of board data
    # print((homographies[0]).shape)        #  (3,3)


    # 3. Closed form solution (Intrinsic, Extrinsic and Distortion coeffs)
    
    intrinsic, B = solve_intrinsic(homographies)
    
    extrinsic = solve_extrinsic(intrinsic, homographies)
    
    
    # print("intrinsic", intrinsic.shape)     # (3, 3)
    # print("extrinsic", extrinsic.shape)     # (3, 4)

    # 4. Non-linear optimization

    #TODO: need to implement (currnet GPT based code is not working well)
    # refined_intrinsic, refined_extrinsic = nonlinear_optimization(intrinsic, extrinsic, objpoints, imgpoints, max_iter=10, tol=1e-6)



# TODO: arrange this below visualization code into a single function
# TODO: modify visualized camera rectangles appropriate to input image resolution and camera size
# show the camera extrinsics
print('Show the camera extrinsics')
# print(extrinsic)
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = intrinsic 
cam_width =   0.064/0.1           #  0.064/0.1
cam_height =   0.032/0.1           #  0.032/0.1
scale_focal =   1600            # 1600
# chess board setting
board_width = 29.7             # 8
board_height = 21            # 6
square_size = 1             # 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = util.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsic, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()



#test
#test2 
