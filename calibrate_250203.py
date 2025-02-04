import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
import os
import util as util
from cv2 import aruco
import scipy.optimize 
import math
def calibrate():
    
    pass
    

def calc_v(h,i,j):
    hi=h[:,i]
    hj=h[:,j]

    vij=[hi[0]*hj[0],hi[0]*hj[1]+hi[1]*hj[0],hi[1]*hj[1],hi[2]*hj[0]+hi[0]*hj[2],hi[2]*hj[1]+hi[1]*hj[2],hi[2]*hj[2]]
    vij=np.array(vij)

    return vij

def find_intrinsic(H):
    V=[]

    for h in H:
        v12=calc_v(h,0,1)
        v11=calc_v(h,0,0)
        v22=calc_v(h,1,1)

        V.append(v12)
        V.append((v11-v22))

    V=np.array(V).reshape(-1,6)

    U,s,VT=np.linalg.svd(V)
    sorted_idx=np.argsort(s)
    b=VT[sorted_idx[0]]

    if(b[0] < 0 or b[2] < 0 or b[5] < 0):
        b = -b
    
    B=np.array([b[0],b[1],b[3],b[1],b[2],b[4],b[3],b[4],b[5]]).reshape(3,3)

    # L = np.linalg.cholesky(B)
    # K = np.linalg.inv(L).transpose() * L[2,2]

    cy=(B[0][1]*B[0][2]-B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]**2)
    scale_factor=(B[2][2]-(B[0][2]**2+cy*(B[0][1]*B[0][2]-B[0][0]*B[1][2]))/B[0][0])
    f_x=math.sqrt(scale_factor/B[0][0])
    f_y=math.sqrt((scale_factor*B[0][0])/(B[0][0]*B[1][1]-B[0][1]**2))
    skew=-B[0][1]*f_x**2*f_y/scale_factor
    cx=skew*cy/f_y-B[0][2]*f_x**2/scale_factor

    intrinsic=np.array([f_x,skew,cx,0,f_y,cy,0,0,1]).reshape(3,3)

    return intrinsic

def find_extrinsic(H,intrinsic):
    R=[]
    T=[]
    extrinsic=[]

    for h in H:
        h1=h[:,0]
        h2=h[:,1]
        h3=h[:,2]

        scale=1/np.linalg.norm(np.linalg.inv(intrinsic)@h1)

        r1=scale*np.linalg.inv(intrinsic)@h1
        r2=scale*np.linalg.inv(intrinsic)@h2
        r3=np.cross(r1,r2)
        t=scale*np.linalg.inv(intrinsic)@h3

        R.append(np.array([r1,r2,r3]).transpose())
        T.append([t])

        extrinsic.append(np.array([r1,r2,r3,t]).transpose())

    return R, T




def reprojection_error_homography(H, objpoints, imgpoints):
    """
    Compute reprojection error for Homography estimation.
    
    Args:
        H (np.array): Homography matrix (3x3)
        objpoints (np.array): 3D world points in homogenous coordinates (N, 3)
        imgpoints (np.array): Corresponding 2D image points (N, 2)
    
    Returns:
        float: Mean reprojection error
    """
    # Convert objpoints to homogeneous coordinates (N, 3) -> (N, 3, 1)
    objpoints_homo = np.hstack((objpoints[:, :2], np.ones((objpoints.shape[0], 1))))  # (N, 3)
    
    # Project using Homography
    projected = (H @ objpoints_homo.T).T  # (N, 3)
    
    # Normalize homogeneous coordinates
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    projected_2d = projected[:, :2]  # Extract (x, y)
    
    # Compute Euclidean distance error
    errors = np.linalg.norm(imgpoints - projected_2d, axis=1)
    mean_error = np.mean(errors)
    
    return mean_error, projected_2d


def solve_extrinsic(intrinsic, homographies):
    extrinsics = []
    K_inv = np.linalg.inv(intrinsic)

    for H in homographies:
        h1, h2, h3 = H[:,0], H[:,1], H[:,2]

        lamda1 = 1.0 / (np.linalg.norm(K_inv @ h1) + 1e-12)
        lamda2 = 1.0 / (np.linalg.norm(K_inv @ h2) + 1e-12)
        lamda3 = (lamda1 + lamda2)*0.5

        r1 = lamda1 * (K_inv @ h1)
        r2 = lamda2 * (K_inv @ h2)
        r3 = np.cross(r1, r2)

        # --- 직교화 ---
        R_tilde = np.column_stack((r1, r2, r3))
        U, _, Vt = np.linalg.svd(R_tilde)
        R = U @ Vt
        # det(R) = -1인 경우 반사를 일으키므로, -1 처리
        if np.linalg.det(R) < 0:
            U[:,-1] *= -1
            R = U @ Vt

        t = lamda3 * (K_inv @ h3)

        # extrinsic: shape (3,4) = [R | t]
        extrinsic_i = np.hstack((R, t.reshape(-1,1)))
        extrinsics.append(extrinsic_i)

    return np.array(extrinsics)  # shape (N,3,4)

def solve_homography(P, m):

    A = []  

    for r in range(len(P)):
        A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0][0], P[r,1]*m[r,0][0], m[r,0][0]])
        A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,0][1], P[r,1]*m[r,0][1], m[r,0][1]])
    
    u, s, vt = np.linalg.svd(A)
    H = np.reshape(vt[-1], (3,3))
    H = H / H[2,2]
    # H = (1/H.item(-1)) * H
    return H

def solve_intrinsic(homographies):
    """
    Use homographies to find out the intrinsic matrix K  

    Return: 
        K: intrisic matrix
        B: B = K^(-T) * K^(-1) 
    """

    # for i, H in enumerate(homographies[:3]):  # 첫 3개만 출력
    #     print(f"Homography {i}:\n", H)

    v = []
    for i in range(len(homographies)):
        h = homographies[i]
        v.append(v_pq(0, 1, h))
        v.append(np.subtract(v_pq(0, 0, h), v_pq(1, 1, h)))
    
    v = np.array(v)
    # print("V matrix: \n", v)
    u, s, vh = np.linalg.svd(v)
    b = vh[-1]

    # Make sure that B is positive definite
    if(b[0] < 0 or b[2] < 0 or b[5] < 0):
        b = -b
   
    B = [[b[0], b[1], b[3]],
         [b[1], b[2], b[4]],
         [b[3], b[4], b[5]]]
    B = np.array(B)
    
    B = B / B[2, 2]                             
    print("B matrix: \n", B)

    # if np.any(np.linalg.eigvals(B) <= 0):
    #     print("Warning: Matrix B is not positive definite! Using SVD instead.")
    #     u, s, vh = np.linalg.svd(B)
    #     B_inv = np.dot(vh.T, np.dot(np.diag(1/s), u.T))  # B의 역행렬
    #     K = np.linalg.inv(np.linalg.cholesky(B_inv).T)
    #     return K, B
  
    # # Solve K by cholesky factoriztion where B=K^(-T)K^(-1), L=K^(-T)
    # L = np.linalg.cholesky(B)
    # K = np.linalg.inv(L).transpose() * L[2,2]
    
    # return K, B

    # Step 6: Check if B is positive definite
    eigvals = np.linalg.eigvals(B)
    if np.any(eigvals <= 0):
        print("⚠ Warning: B matrix is not positive definite! Using SVD instead.")
        u, s, vh = np.linalg.svd(B)
        B = u @ np.diag(np.abs(s)) @ vh  # Ensure positive definite B

    # Step 7: Compute Intrinsic Matrix K
    try:
        L = np.linalg.cholesky(B)
        K = np.linalg.inv(L).T * L[2, 2]
        K /= K[2,2]
    except np.linalg.LinAlgError:
        print("⚠ Warning: Cholesky decomposition failed, using pseudo-inverse.")
        K = np.linalg.pinv(B)

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

def detectchessboard(num_corner_x, num_corner_y, square_size, dir_chess, dir_result):
    
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    dir_board_corner = os.path.join(dir_result, "chessboard_corner")
    if not os.path.exists(dir_board_corner):
        os.makedirs(dir_board_corner)

    # initialize the world coordinates for chessboard
    objp = np.zeros((num_corner_x*num_corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corner_x, 0:num_corner_y].T.reshape(-1,2)  #  (num_corner_x * num_corner_y , 3)
    objp[:, :2] *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(f'{dir_chess}/*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)

        ret, corners = cv2.findChessboardCorners(gray, (num_corner_x,num_corner_y), None)
        print("try")
        if ret == True:
            print("calculate")
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
                obj_text = f"{(objp[i, 0])}, {(objp[i, 1])}, {(objp[i, 2])}"
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

    images = glob.glob(f'{dir_aruco}/*.png')

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
                # cv2.putText(
                #     img,
                #     f"({point2D[0]}, {point2D[1]})",
                #     (point2D[0], point2D[1] + 11),
                #     cv2.FONT_ITALIC,
                #     0.5,
                #     (255, 0, 255),
                #     1,
                #     cv2.LINE_AA,
                # )
                
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

def compute_jacobian(K, distortion_coeff, R, t, projected_points_j, imgpoints_i_j,reprojected_points, image_total_error):
    u0 = K[0,2]
    v0 = K[1,2]
    
    k1 = distortion_coeff[0]
    k2 = distortion_coeff[1]

    #jacobian
    x = projected_points_j[0] / projected_points_j[2]
    y = projected_points_j[1] / projected_points_j[2]
    r = np.sqrt(x**2 + y**2) #radius of distortion

    #observed image coordinates
    # print(np.array(imgpoints_i)[0], np.array(imgpoints_i)[1])
    imgpoints_i_j = np.array([imgpoints_i_j[0], imgpoints_i_j[1], 1], dtype='float').reshape(3,1)  

    #projected image coordinates
    Rt = np.column_stack((R,t))
    uvw = np.dot(  np.dot(K, Rt),  np.hstack((projected_points_j, 1))  )

    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    # u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
    # v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)
    distorted_r = 1 + k1 * r**2 + k2 * r**4
    u_dash = x * distorted_r
    v_dash = y * distorted_r
    reprojected_points.append([float(u_dash), float(v_dash)])
    imgpoints_i_j_dash = np.array([float(u_dash), float(v_dash), 1], dtype='float').reshape(3,1)

    e = np.linalg.norm((imgpoints_i_j - imgpoints_i_j_dash), ord=2)
    image_total_error = image_total_error + e

    return reprojected_points, image_total_error


# def lossFunc(x0, extrinsic, imgpoints, objpoints):
#     """
#     Compute the loss for optimization.
#     """

#     intrinsic = np.array([0,0,0,0,0,0,0,0,1]).reshape(3,3)
#     intrinsic[0,0], intrinsic[0,1], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2], distortion_coeff[0], distortion_coeff[1] = x0

#     u0 = intrinsic[0,2]
#     v0 = intrinsic[1,2]
    
#     k1 = distortion_coeff[0]
#     k2 = distortion_coeff[1]
#     error = []


#     for i in range(len(extrinsic)):
#         R = extrinsic[i][:, :3]
#         t = extrinsic[i][:, 3]
#         Rt = np.column_stack((R,t))
        

#         image_total_error = 0.0

#         objpoints_i = objpoints[i]   
#         imgpoints_i = np.squeeze(imgpoints[i], axis=1)   

#         projected_points = project_points(intrinsic, R, t, objpoints_i)

#         for j in range(len(projected_points)):  
#             imgpoints_i_j = imgpoints_i[j]
#             projected_points_j = projected_points[j]
#             projected_points_j_3d = np.array([projected_points_j[0], projected_points_j[1], 0, 1]).reshape(4,1)
#             # print(projected_points_j.shape) #(3,)
            
#             # 방사 왜곡 보정 (수정된 방식)
#             XYZ = np.dot(Rt, projected_points_j_3d)
#             x = XYZ[0] / XYZ[2]
#             y = XYZ[1] / XYZ[2]
#             r = np.sqrt(x**2 + y**2)

#             uvw = np.dot(  np.dot(intrinsic, Rt),  np.hstack((projected_points_j, 1))  )
#             u = uvw[0] / uvw[2]
#             v = uvw[1] / uvw[2]

#             # u_dash = u + (u - u0) * (k1 * r**2 + k2 * r**4)
#             # v_dash = v + (v - v0) * (k1 * r**2 + k2 * r**4)

#             distorted_r = 1 + k1 * r**2 + k2 * r**4
#             u_dash = (u - u0) * distorted_r
#             v_dash = (v - v0) * distorted_r

#             imgpoints_i_j_dash = np.array([float(u_dash), float(v_dash), 1], dtype='float').reshape(3,1)

#             e = np.linalg.norm((imgpoints_i_j - imgpoints_i_j_dash), ord=2)
#             image_total_error = image_total_error + e

#         error.append(image_total_error/len(projected_points))
#     return np.array(error) / len(extrinsic)


def lossFunc(x0, imgpoints, objpoints, num_images):
    """
    Compute the loss for optimization.
    """
    K = np.array(x0[:9]).reshape(3, 3)
    distortion_coeff = np.array(x0[9:14]).reshape(1, 5)
    extrinsic_params = np.array(x0[14:]).reshape(num_images, 3, 4)
        
    error = []
    
    for i in range(num_images):
        R = extrinsic_params[i][:, :3]  # Rotation matrix (3x3)
        t = extrinsic_params[i][:, 3]  # Translation vector (3x1)

        objpoints_i = np.array(objpoints[i], dtype=np.float32)  # 3D points
        imgpoints_i = np.squeeze(imgpoints[i])  # 2D points

        projected_points, _ = cv2.projectPoints(objpoints_i, R, t, K, distortion_coeff)
        projected_points = projected_points.reshape(-1, 2)

        # L2 norm 오류 계산
        error.append(np.sqrt(np.mean(np.sum((imgpoints_i - projected_points) ** 2, axis=1))))

    return np.hstack(error)  # 모든 오류 값을 1D array로 반환
    

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
    objpoints_homo = np.hstack((objpoints, np.ones((objpoints.shape[0], 1))))  # Homogeneous coordinates (N, 4)
    projected = K @ extrinsic @ objpoints_homo.T  # (3, N)

    # Normalize to get pixel coordinates
    projected /= projected[2, :]  # Normalize by w (Homogeneous normalization)
    projected = projected[:2, :].T  # Extract only (u, v)

    return projected  # Shape: (N, 2)


def compute_reprojection_error(K, extrinsics, objpoints, imgpoints, distortion_coeff):
    error_mat = []

    distortion_coeff = np.array(distortion_coeff, dtype=np.float32).reshape(1, -1)

    for i in range(len(extrinsics)):
        R = extrinsics[i][:, :3]
        t = extrinsics[i][:, 3]

        objpoints_i = np.array(objpoints[i], dtype=np.float32)  # 3D points
        imgpoints_i = np.squeeze(imgpoints[i])  # 2D points

        projected_points, _ = cv2.projectPoints(objpoints_i, R, t, K, distortion_coeff)

        projected_points = projected_points.reshape(-1, 2)

        error = np.sqrt(np.mean(np.sum((imgpoints_i - projected_points) ** 2, axis=1)))
        error_mat.append(error)

    error_average = np.mean(error_mat)

    return error_average, projected_points


def error_from_nonlinear_opt(K_init, K, extrinsics_init, extrinsic, objpoints, imgpoints, homographies, distortion_coeff_init, distortion_coeff): # distortion coeff, intrinsic ..

    prev_error, _ = compute_reprojection_error(K_init, extrinsics_init, objpoints, imgpoints, distortion_coeff_init)
    current_error, points = compute_reprojection_error(K, extrinsic, objpoints, imgpoints, distortion_coeff)

    print("prev_error     :", prev_error)
    print("current_error  :", current_error)

    return extrinsic, points

def loadImages(folder_name):
    files = os.listdir(folder_name)
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

if __name__ == "__main__":
    
    # Initialize settings
    parser = argparse.ArgumentParser(description='calibration arguments')
    parser.add_argument('--mode', default="single", help='choose camera type: single or stereo')
    parser.add_argument('--board_type', default="chessboard", help='select board type: chessboard, arucoboard, charucoboard') 
    parser.add_argument('--result_dir', default='./result', help='direction for result') 
    parser.add_argument('--reproj_result_dir', default='./result/reproj', help='direction for reprojection_result') 

    # chessboard settings
    parser.add_argument('--chess_square_size', default=2.5, help='square size for chessboard in cm') 
    parser.add_argument('--corners_x', default=7, help='input chessboard points in x direction') 
    parser.add_argument('--corners_y', default=7, help='input chessboard points in y direction') 
    parser.add_argument('--chess_data_dir', default='./data/chess_data', help='direction for chessboard data') 

    # arucoboard settings
    parser.add_argument('--aruco_data_dir', default='./data/Cam_001', help='direction for arucoboard data') 
    parser.add_argument('--marker_len', default=7.3, help='marker length in cm') 
    parser.add_argument('--marker_sep_len', default=9.8, help='marker seperation length in cm') 
    parser.add_argument('--squares_vertical', default=5, help='number of squares in vertical') 
    parser.add_argument('--squares_horizontal', default=5, help='number of squares in horizontal') 
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

    # charucoboard settings
    parser.add_argument('--charuco_data_dir', default='./data/charuco_data', help='direction for charucoboard data') 
    parser.add_argument('--charuco_marker_len', default=1.5, help='marker length in cm') 
    parser.add_argument('--charuco_square_len', default=2, help='marker seperation length in cm') 
    parser.add_argument('--charuco_squares_vertical', default=8, help='number of squares in vertical') 
    parser.add_argument('--charuco_squares_horizontal', default=11, help='number of squares in horizontal') 
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
            - imgpoints: 2D points (list of shape: (N_images))
                - Each element in the list is a numpy array of shape: (num_points, 1, 2)
            - objpoints: 3D points (list of shape: (N_images))
                - Each element in the list is a numpy array of shape: (num_points, 3)
    """
    if args.board_type == "chessboard":
        objpoints, imgpoints = detectchessboard(args.corners_x, args.corners_y, args.chess_square_size, args.chess_data_dir, args.result_dir)
        folder_name = args.chess_data_dir
    elif args.board_type == "arucoboard":
        objpoints, imgpoints = detectarucoboard(args.aruco_data_dir, aruco_dict, args.result_dir, args.marker_len, 
                                                args.marker_sep_len, args.squares_vertical, args.squares_horizontal)
        folder_name = args.aruco_data_dir
    elif args.board_type == "charucoboard":
        objpoints, imgpoints = detectcharucoboard(args.charuco_data_dir, charuco_dict, args.result_dir, args.charuco_square_len, 
                                                args.charuco_marker_len, args.charuco_squares_vertical, args.charuco_squares_horizontal)
        folder_name = args.charuco_data_dir
    else:
        print("You set the wrong board type!")

    # 2. Homography estimation (world-image relationship)

    homographies_init = []

    imgpoints_new = imgpoints.copy()

    for i in range(len(imgpoints)):

        imgpoints_new[i] = np.squeeze(imgpoints_new[i])
        # H, mask = cv2.findHomography(objpoints[i][:, :2], imgpoints_new[i], cv2.RANSAC, 5.0) 
        H_manual = solve_homography(objpoints[i], imgpoints[i])

        objpoints_2d = np.array(objpoints[i])[:, :2]
        num_points = objpoints_2d.shape[0]

        objpoints_homo = np.hstack((objpoints_2d, np.ones((num_points, 1))))  # (N,3)
        projected_points_homo = (H_manual @ objpoints_homo.T).T  # (N,3)
        reprojected_points = projected_points_homo[:, :2] / projected_points_homo[:, 2, np.newaxis]  # (N,2)
        error = np.linalg.norm(reprojected_points - imgpoints_new[i], axis=1).mean()
        
        print(f"Homography reprojection error for {i}th image: {error:.2f} pixels")

        homographies_init.append(H_manual)

    # 3. Closed form solution (Intrinsic, Extrinsic and Distortion coeffs)
    
    intrinsic_init, B_init = solve_intrinsic(homographies_init)
    intrinsic_init_mj =find_intrinsic(homographies_init)

    # print(intrinsic_init)
    # print("\n")
    # print(intrinsic_init_mj)
    
    rotation,translation=find_extrinsic(homographies_init,intrinsic_init)
    # extrinsic_init = solve_extrinsic(intrinsic_init, homographies_init)

    # 변환하여 extrinsic_init과 같은 (N, 3, 4) 형태로 만들기
    extrinsic_init_mj = np.zeros((len(rotation), 3, 4))

    for i in range(len(rotation)):
        extrinsic_init_mj[i, :, :3] = rotation[i]  # 3x3 회전 행렬
        extrinsic_init_mj[i, :, 3] = np.array(translation[i]).reshape(-1)  # 리스트 → NumPy 배열로 변환 후 (3,) 형태로 변경

    
    print("extrinsic_init_mj shape:", extrinsic_init_mj.shape)  # (N, 3, 4)
    print(extrinsic_init_mj)  # 첫 번째 변환 행렬 출력

    extrinsic_init = extrinsic_init_mj
    

    # print("intrinsic", intrinsic)     # (3, 3)
    # print("extrinsic", extrinsic)     # (3, 4)
    
    distortion_coeff_init = np.array([[-0.3, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)  # (1,2) 형태 유지
    distortion_coeff = distortion_coeff_init.copy()

    # print("Initial K:\n", intrinsic_init)
    # print("Initial D:\n", distortion_coeff_init)
    # print("Initial extrinsics:\n", extrinsic_init[:3])  # 첫 3개만 출력

    # 4. Non-linear optimization
    x0 = np.hstack([
        intrinsic_init.flatten(),  # 3x3 -> 9개
        distortion_coeff_init.flatten(),  # 5개
        extrinsic_init.flatten()  # (N, 3, 4) → (N * 12)
    ])

    res = scipy.optimize.least_squares(fun=lossFunc, x0=x0, method="trf", args=[imgpoints, objpoints, len(extrinsic_init)], max_nfev=100000)

    optimized_intrinsic = np.array(res.x[:9]).reshape(3, 3)  # (3x3)
    optimized_distortion = np.array(res.x[9:14]).reshape(1, 5)  # (5,)
    optimized_extrinsic = np.array(res.x[14:]).reshape(len(extrinsic_init), 3, 4)  # (N, 3, 4)

    print("\n")
    print("intrinsics ssss", intrinsic_init, "---\n", optimized_intrinsic)
    print("extrincs ssss", extrinsic_init[1],"---\n", optimized_extrinsic[1])

    refined_extrinsic, points = error_from_nonlinear_opt(intrinsic_init, optimized_intrinsic, extrinsic_init, optimized_extrinsic, objpoints, imgpoints, homographies_init, distortion_coeff_init, distortion_coeff)

    ret, K_cv, D_cv, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (3916, 2936), None, None)

    print("OpenCV Calculated Intrinsic Matrix:\n", K_cv)
    print("OpenCV Calculated Distortion Coefficients:\n", D_cv)

    print(rvecs, tvecs)

    def compute_opencv_reprojection_error(K, D, rvecs, tvecs, objpoints, imgpoints):
        total_error = 0
        
        for i in range(len(objpoints)):
            
            objpoints_i = np.array(objpoints[i], dtype=np.float32)  # 3D points
            imgpoints_i = np.squeeze(imgpoints[i])  # 2D points

            projected_points, _ = cv2.projectPoints(objpoints_i, rvecs[i], tvecs[i], K, D)

            error = np.sqrt(np.mean(np.sum((imgpoints_i - projected_points) ** 2, axis=1)))
            total_error += error
        return total_error / len(objpoints)

    opencv_error = compute_opencv_reprojection_error(K_cv, D_cv, rvecs, tvecs, objpoints, imgpoints)
    print(f"OpenCV Baseline Reprojection Error: {opencv_error:.2f} pixels")



    K = np.array(optimized_intrinsic, np.float32).reshape(3,3)
    D = np.array([float(distortion_coeff[0]), float(distortion_coeff[1]), 0, 0] , np.float32)

    save_folder = args.reproj_result_dir
    images = loadImages(folder_name)



    for i, img in enumerate(images):

        # # #intrinsic 점검  --> 왜곡계수에 이상
        # height, width = img[0].shape[:2]
        # _, K_cv, D_cv, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        # print("OpenCV Intrinsic Matrix:\n", D_cv)
        # print("Computed Intrinsic Matrix:\n", D)

        undistorted_img = cv2.undistort(img, K, D)

        # 해당 이미지의 포인트들 가져오기 (예: points[i] 형태가 아니라면 조정 필요)
        points_i = points[i]
        image_points = np.array(points_i, dtype=np.float32).reshape(-1, 1, 2)  # (N,1,2)로 변환

        # 포인트들을 보정된 좌표로 변환
        new_points = cv2.undistortPoints(image_points, K, D)  # 보정된 좌표는 정규화된 카메라 좌표

        # 정규화된 카메라 좌표를 다시 픽셀 좌표로 변환
        new_points = cv2.perspectiveTransform(new_points, K)  # K를 사용하여 픽셀 좌표로 변환
        new_points = new_points.reshape(-1, 2)  # (N,2) 형태로 변환

        # 변환된 포인트 시각화
        for j, (x, y) in enumerate(new_points):
            x, y = int(x), int(y)

            # # 음수 좌표는 이미지 범위를 벗어나므로 스킵
            # if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            #     print(f"Skipping point {j} ({x}, {y}) out of bounds")
            #     continue

            # 보정된 이미지에 점 표시
            undistorted_img = cv2.circle(undistorted_img, (x, y), 16, (0, 0, 255), -1)

        # 저장
        filename = os.path.join(save_folder, f"{i}_reproj.png")
        cv2.imwrite(filename, undistorted_img)
        print(f"Saved: {filename}")

    print("Processing complete!")









# # TODO: arrange this below visualization code into a single function
# # TODO: modify visualized camera rectangles appropriate to input image resolution and camera size
# # show the camera extrinsics
# print('Show the camera extrinsics')
# # print(extrinsic)
# # plot setting
# # You can modify it for better visualization
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# # camera setting
# camera_matrix = intrinsic 
# cam_width =   0.064/0.1        #  0.064/0.1
# cam_height =   0.032/0.1       #  0.032/0.1
# scale_focal =   1600           # 1600
# # chess board setting
# board_width = 20             # 8
# board_height = 15              # 6
# square_size = 1.5                # 1
# # display
# # True -> fix board, moving cameras
# # False -> fix camera, moving boards
# min_values, max_values = util.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
#                                                 scale_focal, extrinsic, board_width,
#                                                 board_height, square_size, True)

# X_min = min_values[0]
# X_max = max_values[0]
# Y_min = min_values[1]
# Y_max = max_values[1]
# Z_min = min_values[2]
# Z_max = max_values[2]
# max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

# mid_x = (X_max+X_min) * 0.5
# mid_y = (Y_max+Y_min) * 0.5
# mid_z = (Z_max+Z_min) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, 0)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ax.set_xlabel('x')
# ax.set_ylabel('z')
# ax.set_zlabel('-y')
# ax.set_title('Extrinsic Parameters Visualization')
# # plt.show()

