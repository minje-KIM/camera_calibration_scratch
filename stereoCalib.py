import cv2
import numpy as np
from scipy.linalg import svd
import glob
import os
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
markersize=98

def file_glob(path):
    flist = []
    for ext in extensions:
        flist.extend(glob.glob(os.path.join(path, ext)))
    return flist

def solveHomography(objp,imgp):
    '''
    objp: object point in 3D world coordinate; 2D point (Z=0) [Nx2]
    imgp: 2D img point (corner of checkerboard) [Nx2]
    '''
    A=np.ones(9)
    for i in range(len(objp)):
        Ai=[]
        Ai.append(np.zeros((3)))
        Ai.append(-imgp[i][2]*objp[i])
        Ai.append(imgp[i][1]*objp[i])
        Ai.append(imgp[i][2]*objp[i])
        Ai.append(np.zeros((3)))
        Ai.append(-imgp[i][0]*objp[i])

        Ai=np.array(Ai).reshape(2,-1)
    
        A=np.vstack((A,Ai))
    
    return A[1:,:]

def findHomography(objp,imgp):
    objp=objp[:,np.newaxis,:2]
    imgp=imgp[:,np.newaxis,:2]

    H,_=cv2.findHomography(objp,imgp,cv2.RANSAC)

    return H

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

    U,s,VT=svd(V)
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

def homography_viz(homography,imgp,objp,img):
    '''
    project 3D world point to 2D image using estimated homography
    '''

    for i in range(len(imgp)):
        cv2.circle(img,(int(imgp[i][0]),int(imgp[i][1])),radius=10,color=(0,0,255),thickness=-1)
        
        proj=homography@objp[i]
        proj[0]=proj[0]/proj[2]
        proj[1]=proj[1]/proj[2]

        cv2.circle(img,(int(proj[0]),int(proj[1])),radius=10,color=(255,0,0),thickness=-1)

    cv2.imwrite('E2VID/CameraCalibration-FromScratch/corner_data/img.png',img)

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

def rmat2rvec(rotation):
    rvecs=[]

    for rmat in rotation:
        rvec=R.from_matrix(rmat)
        rvec=rvec.as_rotvec()

        rvecs.append(rvec)

    return rvecs

def calculate_reprojection_error(param,filelist,imgps,objps):

    f_x=param[0]
    skew=param[1]
    # skew=0
    cx=param[2]
    f_y=param[3]
    cy=param[4]

    intrinsic=np.array([f_x,skew,cx,0,f_y,cy,0,0,1]).reshape(3,3)

    reprojection_error=[]
    rmse=0

    for i, file in enumerate(filelist):
        imgp=imgps[i]
        objp=objps[i]

        rvec=param[i*6+5:i*6+8]
        tvec=param[i*6+8:i*6+11]
        dist=param[-2:]

        rmat=R.from_rotvec(rvec)
        rmat=rmat.as_matrix()

        for i,point in enumerate(objp):
            w2c=rmat@point+tvec
            camx=w2c[0]/w2c[2]
            camy=w2c[1]/w2c[2]

            r2=camx**2+camy**2
            r4=r2**2

            r_coeff=1+dist[0]*r2+dist[1]*r4

            imgx=camx*r_coeff*f_x+skew*camy*r_coeff+cx
            imgy=camy*r_coeff*f_y+cy

            residual=[imgp[i][0]-imgx,imgp[i][1]-imgy]

            reprojection_error+=residual

        proj,_=cv2.projectPoints(objp,rvec,tvec,intrinsic,np.concatenate((dist,[0,0,0])))
        rmse+=mean_squared_error(proj.squeeze(),imgp[:,:2])

    # rmse
    rmse=rmse/len(filelist)
    print(rmse**0.5)
    # print(np.mean(np.abs(np.array(reprojection_error))))

    return np.array(reprojection_error).flatten()

def refine_param(filelist,imgPoint,objPoint,intrinsic,rvecs,tvecs,dists):
    f_x=intrinsic[0][0]
    skew=intrinsic[0][1]
    # skew=0
    cx=intrinsic[0][2]
    f_y=intrinsic[1][1]
    cy=intrinsic[1][2]

    param=np.array([f_x,skew,cx,f_y,cy])

    for i, rvec in enumerate(rvecs):
        param=np.concatenate([param,rvec,tvecs[i][0]])
    param=np.concatenate([param,dists])

    reprojection_error=calculate_reprojection_error(param,filelist,imgPoint,objPoint)

    new_param=least_squares(calculate_reprojection_error,param,method='lm',ftol=1e-08,xtol=1e-08,gtol=1e-08,args=(filelist,imgPoint,objPoint),verbose=1)

    f_x=new_param['x'][0]
    skew=new_param['x'][1]
    cx=new_param['x'][2]
    f_y=new_param['x'][3]
    cy=new_param['x'][4]

    intrinsic_refine=np.array([f_x,skew,cx,0,f_y,cy,0,0,1]).reshape(3,3)

    rvec_refine=[]
    tvec_refine=[]
    dist_refine=[]

    for i in range(len(rvecs)):
        rvec_refine.append(new_param['x'][i*6+5:i*6+8])
        tvec_refine.append(new_param['x'][i*6+8:i*6+11])
    dist_refine.append(new_param['x'][-2:])

    return intrinsic_refine,rvec_refine,tvec_refine,dist_refine
import cv2
import numpy as np

def detect_charuco_corners(image_path, output_path):
    """
    Charuco 보드의 코너를 감지하고, 원본 이미지에 빨간 점으로 오버랩하여 저장

    Args:
        image_path (str): 원본 이미지 파일 경로.
        output_path (str): 코너가 표시된 이미지를 저장할 경로.
    """
    # ArUco 및 Charuco 보드 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

    # 사용자 정의 Charuco 보드 설정 (6x9, 사각형 9.8cm, 마커 7.3cm)
    number_x_square = 5
    number_y_square = 5
    length_square = markersize  # 사각형 크기 (미터 단위)
    length_marker = 73  # 마커 크기 (미터 단위)

    board = cv2.aruco.CharucoBoard((number_x_square,number_y_square), length_square,length_marker, aruco_dict)
    # charuco_board = cv2.aruco.CharucoBoard_create(
    #     number_x_square, number_y_square, length_square, length_marker, aruco_dict
    # )

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ArUco 마커 탐지
    parameters=cv2.aruco.DetectorParameters()
    corners, ids, _ =cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        # Charuco 코너 보정 및 감지
        charucodetector=cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(image)
        # _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        #     corners, ids, gray, charuco_board)

        # Charuco 코너를 원본 이미지에 빨간 점으로 오버랩
        if charuco_corners is not None:
            for corner in charuco_corners:
                x, y = int(corner[0][0]), int(corner[0][1])  # 2D 좌표
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원

        # 결과 이미지 저장
        cv2.imwrite(output_path, image)
        print(f"Output saved to {output_path}")
    else:
        print("No ArUco markers detected in the image.")

    return charuco_corners

def singleCalib(path,markersize):
    
    filelist = sorted(file_glob(path))

    imgPoint = []
    objPoint = []
    Homography = []

    width=4
    height=4

    for file in filelist:

        fname=file[-9:-4]

        save_path=os.path.join('E2VID/CameraCalibration-FromScratch/corner_data',fname+'_corner.png')
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # define object point (3D world)

        objp = np.zeros((width*height,3),np.float32)
        objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)

        objp*=markersize
        objp[:,2]=1

        imgp=[]

        corners=detect_charuco_corners(file,save_path)
        
        imgp.append(corners)
        objPoint.append(objp)

        imgp=imgp[0]
        imgp=np.squeeze(imgp)

        aug=np.ones((len(imgp),1))
        imgp=np.concatenate((imgp,aug),axis=1)

        imgPoint.append(np.array(imgp,np.float32))

        A=solveHomography(objp,imgp)
        U, s, VT=svd(A)

        sorted_idx=np.argsort(s)

        H=VT[sorted_idx[0]]
        H=(1/H[-1])*H.reshape(3,3)
        Homography.append(H)

    # calculate intrinsic and extrinsic parameter
    intrinsic=find_intrinsic(Homography)
    rotation,translation=find_extrinsic(Homography,intrinsic)
    
    # convert rotation matrix to rotation vector
    rvecs=rmat2rvec(rotation)
    dist_init=np.array([0,0])

    # refinement (optimization)
    intrinsic_refine,rvec_refine,tvec_refine,dist_refine=refine_param(filelist,imgPoint,objPoint,intrinsic,rvecs,translation,dist_init)

    print('Intrinsic:',intrinsic,'Intrinsic_refine:',intrinsic_refine)
    print('dist_refine:',dist_refine)

    print('done!')

    return filelist,imgPoint,objPoint,intrinsic_refine,rvec_refine,tvec_refine,dist_refine

def refine_extrinsic_param(filelist,objPoint_L,imgPoint_R,intrinsic_R,rvec_L,tvec_L,rvec_R,tvec_R,dist_R):
    param=np.zeros(0)
    for i,rvec in enumerate(rvec_L):
        param=np.concatenate([param,rvec,tvec_L[i],rvec_R[i],tvec_R[i]])
    
    new_param=least_squares(proj_error_L2R,param,method='lm',ftol=1e-08,xtol=1e-08,gtol=1e-08,max_nfev=200,args=(filelist,objPoint_L,imgPoint_R,intrinsic_R,dist_R),verbose=1)

    rvecL_refine=[]
    tvecL_refine=[]
    rvecR_refine=[]
    tvecR_refine=[]
    
    for i in range(len(rvec_L)):
        rvecL_refine.append(new_param['x'][i*12:i*12+3])
        tvecL_refine.append(new_param['x'][i*12+3:i*12+6])
        rvecR_refine.append(new_param['x'][i*12+6:i*12+9])
        tvecR_refine.append(new_param['x'][i*12+9:i*12+12])
    
    return rvecL_refine,tvecL_refine,rvecR_refine,tvecR_refine

def proj_error_L2R(param,filelist,objPoint_L,imgPoint_R,intrinsic_R,dist_R):
        f_x = intrinsic_R[0][0]
        skew=intrinsic_R[0][1]
        cx=intrinsic_R[0][2]
        f_y=intrinsic_R[1][1]
        cy=intrinsic_R[1][2]

        proj_error=[]
        rmse=0

        for i, file in enumerate(filelist):
            proj=[]
            imgp_R=imgPoint_R[i]
            objp_L=objPoint_L[i]

            rvec_L=param[i*12:i*12+3]
            tvec_L=param[i*12+3:i*12+6]
            rvec_R=param[i*12+6:i*12+9]
            tvec_R=param[i*12+9:i*12+12]

            rmat_L=R.from_rotvec(rvec_L)
            rmat_L=rmat_L.as_matrix()

            rmat_R=R.from_rotvec(rvec_R)
            rmat_R=rmat_R.as_matrix()

            for j, objp in enumerate(objp_L):

                cam_L=rmat_L@objp+tvec_L
                rmat_L2R=rmat_R@np.linalg.inv(rmat_L)
                tvec_L2R=tvec_R-rmat_L2R@tvec_L
                cam_L2R=rmat_L2R@cam_L+(tvec_R-rmat_L2R@tvec_L)

                camx_L2R=cam_L2R[0]/cam_L2R[2]
                camy_L2R=cam_L2R[1]/cam_L2R[2]

                r2=camx_L2R**2+camy_L2R**2
                r4=r2**2

                r_coeff=1+dist_R[0][0]*r2+dist_R[0][1]*r4

                imgx=camx_L2R*r_coeff*f_x+skew*camy_L2R*r_coeff+cx
                imgy=camy_L2R*r_coeff*f_y+cy

                residual=[imgp_R[j][0]-imgx,imgp_R[j][1]-imgy]
                proj_error+=residual

                proj.append(np.array([imgx,imgy]))
                
            rmse+=mean_squared_error(proj,imgp_R[:,:2])

        rmse=rmse/len(filelist)
        # rmse
        print(rmse**0.5)

        return np.array(proj_error).flatten()

def transform_L2R(rvecs_L,tvecs_L,rvecs_R,tvecs_R):
    rmats_L2R=[]
    tvecs_L2R=[]

    for i in range(len(rvecs_L)):
        rvec_L=rvecs_L[i]
        tvec_L=tvecs_L[i]

        rvec_R=rvecs_R[i]
        tvec_R=tvecs_R[i]

        rmat_L=R.from_rotvec(rvec_L)
        rmat_L=rmat_L.as_matrix()

        rmat_R=R.from_rotvec(rvec_R)
        rmat_R=rmat_R.as_matrix()

        rmat_L2R = rmat_R@np.linalg.inv(rmat_L)
        tvec_L2R = tvec_R-rmat_L2R@tvec_L

        rmats_L2R.append(rmat_L2R)
        tvecs_L2R.append(tvec_L2R)
    
    return rmats_L2R, tvecs_L2R

def rectified_images(imgL,imgR,img_size,K_L,dist_L,K_R,dist_R,rmats_L2R,tvecs_L2R):
    zeros=np.zeros(3)
    dist_L=np.concatenate((dist_L,zeros))
    dist_R=np.concatenate((dist_R,zeros))

    for i in range(len(tvecs_L2R)):

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L,dist_L,K_R,dist_R,img_size,rmats_L2R,tvecs_L2R)

    f=Q[2,3]
    b=1/Q[3,2]

    K_Lnew=P1[:3,:3]
    K_Rnew=P2[:3,:3]

    mapx_L, mapy_L=cv2.initUndistortRectifyMap(K_L,dist_L,R1,K_Lnew,img_size,cv2.CV_32FC1)
    mapx_R, mapy_R=cv2.initUndistortRectifyMap(K_R,dist_R,R2,K_Rnew,img_size,cv2.CV_32FC1)

    imgL_rect=cv2.remap(imgL,mapx_L,mapy_L,interpolation=cv2.INTER_LINEAR)
    imgR_rect=cv2.remap(imgR,mapx_R,mapy_R,interpolation=cv2.INTER_LINEAR)

    return imgL_rect,imgR_rect,f, b

def draw_horizontal_lines(img, num_lines=10):
    """Draw evenly spaced horizontal lines on an image."""
    img_with_lines = img.copy()
    h, w = img.shape[:2]
    step = h // num_lines

    for y in range(0, h, step):
        cv2.line(img_with_lines, (0, y), (w, y), (0, 0, 255), thickness=4)  # Green lines

    return img_with_lines

def visualize_rectified_images(left_img, right_img,fname):
    """Draw horizontal lines and display both rectified images."""
    left_with_lines = draw_horizontal_lines(left_img)
    right_with_lines = draw_horizontal_lines(right_img)

    # Convert BGR to RGB for matplotlib
    left_with_lines = cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB)
    right_with_lines = cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB)

    # Plot images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(left_with_lines)
    plt.title("Left Rectified Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(right_with_lines)
    plt.title("Right Rectified Image")
    plt.axis("off")

    save_path=os.path.join('/home/minji/stereo_calib/1204_stereo/rectified',fname)
    plt.savefig(save_path)

    plt.cla()
    plt.clf()
    plt.close()

def calculate_y_error(filelist_L,filelist_R,intrinsic_L,dist_L,intrinsic_R,dist_R,rmats_L2R,tvec_L2R,img_size):
    y_error=0

    for i in range(len(filelist_L)):
        imgL = cv2.imread(filelist_L[i])
        imgR = cv2.imread(filelist_R[i])

        imgRect_L, imgRect_R, f, b = rectified_images(imgL,imgR,img_size,intrinsic_L,dist_L[0],intrinsic_R,dist_R[0],rmats_L2R[i],tvec_L2R[i])

        fname=filelist_L[i][-9:-4]
        filename='rectified'+fname+'_rectified.png'

        rect_left=os.path.join('/home/minji/stereo_calib/1204_stereo/rectified/left',filename)
        rect_right=os.path.join('/home/minji/stereo_calib/1204_stereo/rectified/right',filename)

        cv2.imwrite(rect_left, imgRect_L)
        cv2.imwrite(rect_right, imgRect_R)

        visualize_rectified_images(imgRect_L, imgRect_R,filename)

    imgPoint_L=find_corner('/home/minji/stereo_calib/1204_stereo/rectified/left')
    imgPoint_R=find_corner('/home/minji/stereo_calib/1204_stereo/rectified/right')

    cnt=0

    for fname in imgPoint_L.keys():
        if fname in imgPoint_R.keys():
            imgL=imgPoint_L[fname]
            imgR=imgPoint_R[fname]
            
            cnt+=1

            for i in range(len(imgL)):
                residual=np.abs(imgL[i][1]-imgR[i][1])
        
                y_error+=residual

            y_error=y_error/len(imgL)
    
    y_error=y_error/cnt

    return y_error


def refine_param(filelist,imgPoint,objPoint,intrinsic,rvecs,tvecs,dists):
    f_x=intrinsic[0][0]
    skew=intrinsic[0][1]
    cx=intrinsic[0][2]
    f_y=intrinsic[1][1]
    cy=intrinsic[1][2]

    param=np.array([f_x,skew,cx,f_y,cy])

    for i, rvec in enumerate(rvecs):
        param=np.concatenate([param,rvec,tvecs[i][0]])
    param=np.concatenate([param,dists])

    reprojection_error=calculate_reprojection_error(param,filelist,imgPoint,objPoint)

    new_param=least_squares(calculate_reprojection_error,param,method='lm',ftol=1e-08,xtol=1e-08,gtol=1e-08,args=(filelist,imgPoint,objPoint),verbose=1)

    f_x=new_param['x'][0]
    skew=new_param['x'][1]
    cx=new_param['x'][2]
    f_y=new_param['x'][3]
    cy=new_param['x'][4]

    intrinsic_refine=np.array([f_x,skew,cx,0,f_y,cy,0,0,1]).reshape(3,3)

    rvec_refine=[]
    tvec_refine=[]
    dist_refine=[]

    for i in range(len(rvecs)):
        rvec_refine.append(new_param['x'][i*6+5:i*6+8])
        tvec_refine.append(new_param['x'][i*6+8:i*6+11])
    dist_refine.append(new_param['x'][-2:])

    return intrinsic_refine,rvec_refine,tvec_refine,dist_refine

def find_corner(path):
    
    filelist = sorted(file_glob(path))

    imgPoint = dict()

    width=4
    height=4

    for file in filelist:

        fname=file[-19:-14]

        save_path=os.path.join('E2VID/CameraCalibration-FromScratch/corner_data',fname+'rec_corner.png')
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        imgp=[]

        corners=detect_charuco_corners(file,save_path)

        if len(corners)==16:
        
            imgp.append(corners)

            imgp=imgp[0]
            imgp=np.squeeze(imgp)

            imgPoint[fname]=imgp

    return imgPoint

def main():

    left_path=os.path.join('./data/Cam_001')
    right_path=os.path.join('./data/Cam_002')

    filelist_L,imgp_L,objp_L,intrinsic_L,rvec_L,tvec_L,dist_L=singleCalib(left_path,markersize)
    filelist_R,imgp_R,objp_R,intrinsic_R,rvec_R,tvec_R,dist_R=singleCalib(right_path,markersize)

    new_rvecL,new_tvecL,new_rvecR,new_tvecR=refine_extrinsic_param(filelist_R,objp_L,imgp_R,intrinsic_R,rvec_L,tvec_L,rvec_R,tvec_R,dist_R)

    img_size=(1920,1200)

    # Compute rectification maps
    rmats_L2R, tvecs_L2R = transform_L2R(new_rvecL,new_tvecL,new_rvecR,new_tvecR)

    y_error=calculate_y_error(filelist_L,filelist_R,intrinsic_L,dist_L,intrinsic_R,dist_R,rmats_L2R,tvecs_L2R,img_size)

    print(y_error)
    print('Done!')

main()