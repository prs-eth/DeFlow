from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
from sklearn import preprocessing

def qvec2rotmat(qvec):
    """_summary_

    Args:
        qvec (4-d vector): 

    Returns:
        np: [3*3] rotation matrix
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    """_summary_

    Args:
        R (3*3): rotation matrix

    Returns:
        qvec (4d vec): quaternion
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class LiCamera:
    """
    Class for a camera-LiDAR setup
    Includes intrinsics and extrinsics for camera under LiDAR frame
        K: camera intrinsics 3*3 matrix
        dist_coef: distortion parameters. Same format in opencv
        rot: rotation matrix 3*3 from LiDAR to Camera
        t: translation vector 3*1 from LiDAR to Camera
    
    Author: Liyuan Zhu(liyzhu@ethz.ch)
    """
    def __init__(self, camera_path) -> None:
        K, t_vec, rot_mtx, dist_coef = read_camera_params(camera_path)
        self.K = K.astype(np.float32)
        self.rot_mtx = rot_mtx.astype(np.float32)
        self.t_vec = t_vec.astype(np.float32)
        self.dist_coef = dist_coef
        #resize K and image
        self.img_w = 1600
        self.img_h = 960
        self.K[1,2] -= 120.0 
        self.K[0,2] -= 160.0 
        
    def project2image(self, points):
        num = points.shape[0]
        # pts_2d, _ = cv2.projectPoints(points, self.rot_mtx, self.t_vec, self.K, np.zeros_like(self.dist_coef))
        pts_2d = self.K @ (self.rot_mtx @ points.T + self.t_vec)
        pts_2d = pts_2d.T[:,:2] / pts_2d.T[:,2].reshape(num,1)
        # pts_2d = np.squeeze(pts_2d)
        return pts_2d
    
    def FoV_filtering(self, points):
        # remove the 3d points out of camera's FOV
        # return the filtered point clouds
        pts_2d = self.project2image(points)
        points = points[np.where((pts_2d[:,0]>0)&(pts_2d[:,0]<self.img_w)&(pts_2d[:,1]>0)&(pts_2d[:,1]<self.img_h))]
        points = points[points[:,0] != 0]
        return points
    
    def T_world2cam(self, points):
        # transform 3d point cloud from world coordinate to camera coordinate
        pts_3d = self.rot_mtx @ points.T + self.t_vec
        return pts_3d
    
    def T_cam2world(self, points_cam):
        # transform 3d points in camera frame back to world(LiDAR) frame
        points_w = self.rot_mtx.T @ (points_cam - self.t_vec)
        return points_w
    
    def LiDAR_to_depth(self, points):
        """ transform LiDAR data to depth on image
        Returns:
            depth: [N, 3]: N *(x_img, y_img, depth)
        """
        pts_cam = self.T_world2cam(points).T
        pts_2d = self.K @ pts_cam.T
        pts_2d = pts_2d.T[:,:2] / pts_2d.T[:,2].reshape(points.shape[0],1)
        pts_depth = np.concatenate([pts_2d[:,:2], pts_cam[:,[2]]], axis = 1)
        pts_depth = pts_depth[pts_depth[:,2]>0] # remove negative depth indices
        return pts_depth

def read_camera_params(cam_path):
    """read camera in-extrinsics from text

    Args:
        cam_path (str): _description_

    Returns:
        K: 3*3 camera intrinsic matrix
        t_vec: 3*1 translation vector
        rot_mtx: 3*3 rotation matrix from LiDAR to Camera
        dist_coef: camera distortion coefficients(same format as in cv2)
    """
    path_intrinsics = Path(cam_path)/'cam_intrinsics.txt'
    path_transformation = Path(cam_path)/'LiCam_transformation.txt'
    intrinsics, transformation = [], []
    with open(path_intrinsics) as f:
        for line in f:
            line = line.strip('\n')
            intrinsics.append(np.fromstring(line, dtype=np.float64, sep=' '))
    
    with open(path_transformation) as f:
        for line in f:
            line = line.strip('\n')
            transformation.append(np.fromstring(line, dtype=np.float64, sep=' '))

    K = np.concatenate(intrinsics[0:3]).reshape(3, 3)
    dist_coef = np.concatenate((intrinsics[3],np.zeros((2))), axis=0) # radial distortion paramters
    rot_mtx = np.concatenate(transformation[0:3]).reshape(3, 3).T
    t_vec = transformation[3].reshape(3,1)
    
    return K, t_vec, rot_mtx, dist_coef
    

def visualize_Li2Cam(data_path, time_step:int):
    """
    Project 3D LiDAR points onto image 
    Input: 
        camera folder: includes camera intrinsics and camera-LiDAR transformation
        data_path: includes images and lidar scans
        time step: from 0 to 2000
    Author: Liyuan Zhu(liyzhu@ethz.ch)
    """
    # converting time step from int to str
    time_step = f'0{1000+time_step}'
    path_cam1 = Path(data_path)/'Transformations/GazoducCamera1'
    path_cam2 = Path(data_path)/'Transformations/GazoducCamera2'
    
    path_image1 = Path(data_path)/('Data/Cam1/'+time_step+'.jpg')
    path_image2 = Path(data_path)/('Data/Cam2/'+time_step+'.jpg')
    path_scan = Path(data_path)/('Data/PLY_Files/'+time_step+'.ply')
    
    # initialize camera LiDAR setup
    LiCam1 = LiCamera(path_cam1)
    LiCam2 = LiCamera(path_cam2)
    
    #read point cloud and image
    pcd=o3d.io.read_point_cloud(str(path_scan))
    points = np.asarray(pcd.points)
    # points = LiCam1.FoV_filtering(points)
    # points = LiCam2.FoV_filtering(points)
    cam1_pts_2d = LiCam1.project2image(points)
    cam2_pts_2d = LiCam2.project2image(points)
    
    z = preprocessing.normalize([points[:,2]])
    fig = plt.figure(figsize=(12,10),dpi=96,tight_layout=True)
    image1 = mpimg.imread(path_image1)
    IMG_H,IMG_W,_ = image1.shape
    fig.add_subplot(2,1,1)
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(image1)
    plt.scatter([cam1_pts_2d[:,0]],[cam1_pts_2d[:,1]],c=[z],cmap='rainbow_r',alpha=0.5,s=1)
    plt.axis('off')
    #plt.show()
    
    image2 = mpimg.imread(path_image2)
    IMG_H,IMG_W,_ = image2.shape
    fig.add_subplot(2,1,2)
    plt.axis([0,IMG_W,IMG_H,0])
    plt.imshow(image2)
    plt.scatter([cam2_pts_2d[:,0]],[cam2_pts_2d[:,1]],c=[z],cmap='rainbow_r',alpha=0.5,s=1)
    plt.axis('off')
    plt.show()
    


if __name__ == "__main__":
    time_step = 0
    data_path = '/scratch/liyzhu/debris_flow/DF_LiDAR_Data/'
    visualize_Li2Cam(data_path, time_step)
    # LiCam1 = LiCamera(Path(data_path)/'Transformations/GazoducCamera1')