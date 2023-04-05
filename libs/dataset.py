import glob
from tkinter.tix import Tree
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from libs.LiCam import LiCamera
import torch
import open3d as o3d
from PIL import Image

from .augmentation import random_flip, random_crop

def get_train_val_test_split(data_len, path_val, path_test):
    
    val_ids = np.loadtxt(path_val, dtype=np.int32)
    test_ids = np.loadtxt(path_test, dtype=np.int32)

    # make sure no intersection between train and val
    train_ids = [i for i in range(data_len) if i not in val_ids and i not in test_ids and (i+1) not in val_ids and (i+1) not in test_ids]
    train_ids = np.asarray(train_ids)
    
    return train_ids, val_ids, test_ids

class DeFlow_Dataset(Dataset):
    '''
    Class for debris flow dataset
    Input: path to the data
    Author: Liyuan Zhu(liyzhu@ethz.ch)
    Return:
        time_step: [ind, ind+1]
        pointCloud: point clouds in two consecutive steps
        camera1: images from camera L in two steps
        camera2: images from camera R in two steps
        LiCam1: Camera LiDAR geometry for cam1 and LiDAR
        LiCam2: Camera LiDAR geometry for cam2 and LiDAR
    '''
    def __init__(self, args) -> None:
        super().__init__()
        self.data_path = args.dataset.path
        self.lidar_scans = sorted(glob.glob(f'{self.data_path}/Data/PLY_Files/*.ply'))
        self.imagesS = sorted(glob.glob(f'{self.data_path}/Data/Cam1/*.jpg'))
        self.imagesT = sorted(glob.glob(f'{self.data_path}/Data/Cam2/*.jpg'))
        
        self.LiCamS = LiCamera(Path(self.data_path)/'Transformations/GazoducCamera1')
        self.LiCamT = LiCamera(Path(self.data_path)/'Transformations/GazoducCamera2') 
        # bbox to crop the point cloud [xmin,ymin,zmin,xmax,ymax,zmax]
        self.bbox = args.dataset.bbox
        self.min_depth = args.dataset.min_depth
        self.max_depth = args.dataset.max_depth
        # self.data_augmentation = args.data_augmentation
        self._full_len = len(self.imagesT) - 1
        # self.mode = 'train'
        self.augmentation = args.augmentation.enabled
        self.flip_mode = args.augmentation.flip_mode
        self.crop_size = args.augmentation.crop_size
        self.input_depth = args.network.input_depth
        self.train_ids, self.val_ids, self.test_ids = get_train_val_test_split(len(self.imagesT) - 1, args.dataset.val_ids, args.dataset.test_ids)
        self.depth_input_ratio = args.dataset.depth_input_ratio
        
    def __len__(self):
        return len(self.imagesT) - 1
    
    def __getitem__(self, index):
        next_idx = index + 1
        pcd1 = self._get_pointcloud(self.lidar_scans[index])
        pcd2 = self._get_pointcloud(self.lidar_scans[next_idx])
        # filter the point cloud to be only inside the view of camera 2
        pcd1 = self.LiCamT.FoV_filtering(pcd1)
        pcd2 = self.LiCamT.FoV_filtering(pcd2)
        
        # crop point cloud
        # pcd1, pcd2 = self.crop_point_cloud(pcd1), self.crop_point_cloud(pcd2)
        imgT_1 = self._get_image(self.imagesT[index])
        imgT_2= self._get_image(self.imagesT[next_idx])
        # imgS_1= self._get_image(self.imagesS[index])
        # imgS_2 = self._get_image(self.imagesS[next_idx])
        
        h_orig, w_orig = imgT_1.shape[-2:]
        input_im_size = np.array([h_orig, w_orig], dtype=np.float32)
        
        # convert point cloud to sparse depth on image
        # inversing depth here to make sure closer points would have a larger weights in l1_loss
        depth_t1 = self._get_inverse_depth(pcd1)
        depth_t2 = self._get_inverse_depth(pcd2)
        data = {
            'time_step': np.asarray([index, index+1], dtype=np.int32), 
            'imgT_1': imgT_1, # top-view camera T
            'imgT_2': imgT_2,
            'k_T': self.LiCamT.K,
            'rot_T': self.LiCamT.rot_mtx,
            'input_size': input_im_size,
            'depth_t1': depth_t1,
            'depth_t2': depth_t2,
            'flip': False,
            
        }
        
        if self.augmentation == True and index in self.train_ids:
            data = self._augmentation(data)
            # augment the depth input ratio [0.2, 0.8]
            depth_input_ratio = np.random.random_sample() * 0.6 + 0.2
            data = self._random_split_depth(data, depth_input_ratio)
        else:
            data = self._random_split_depth(data, self.depth_input_ratio)
        
        return data

    
    def _get_image(self, img_path, crop = True, resize = False):
        # normalize to [-1, 1]
        
        if crop == True:
            image = np.array(Image.open(img_path), dtype=np.float32)/128 - 1
            image = image[120:, 160:1760]
            # k = self.LiCamT.K
            # k[1,2] -= 120.0 
        
        if resize == True:
            image = Image.open(img_path)
            w, h = image.size
            image = image.resize((w//3*2, h//3*2))
            k = self.LiCamT.K
            k[0] /= resize
            k[1] /= resize
            image = np.array(image, dtype=np.float32)/128 - 1
            
        return np.transpose(image,(2,0,1))
    
    def _get_pointcloud(self, pcd_path):
        pc = np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)
        return pc
    
    def _get_inverse_depth(self, pc, desired_len=20000):
        # transform the point cloud into depth on image
        pc = self.LiCamT.LiDAR_to_depth(pc)
        # filter the depth to [min_depth, max_depth]
        ids = np.logical_and([pc[:,-1]>self.min_depth], [pc[:,-1]<self.max_depth]).squeeze(0)
        pc = pc[ids]
        # inverse depth(pseudo disparity) range [1/80, 1]
        pc[:,-1] = self.min_depth / pc[:,-1]
        
        # pad point cloud to enable batchsize > 1
        return pc
    
    def _random_split_depth(self, data_dict, depth_input_ratio):
        depth1 = self.unpad_point_cloud(data_dict['depth_t1'])
        
        # gen random indices to split the data
        input_ids_1 = np.random.choice(range(depth1.shape[0]), size=int(depth1.shape[0]*depth_input_ratio), replace=False)
       
        input_depth1 = depth1[input_ids_1]
        data_dict['input_depth_t1']  = self._pc_to_image(input_depth1, data_dict['input_size'])
        
        # image 2
        depth2 = self.unpad_point_cloud(data_dict['depth_t2'])
        input_ids_2 = np.random.choice(range(depth2.shape[0]), size=depth2.shape[0]//2, replace=False)
        
        input_depth2 = depth2[input_ids_2]
       
        data_dict['input_depth_t2']  = self._pc_to_image(input_depth2, data_dict['input_size'])
        
        return data_dict
    
    
    def _pc_to_image(self, pc, image_size):
        grid_ids = np.rint(pc[:,:2]).astype(np.int32)
        # safety check no out of range
        mask1 = grid_ids[:,0] < image_size[1]
        mask2 = grid_ids[:,1] < image_size[0]
        mask = mask1 * mask2
        grid_ids = grid_ids[mask]
        pc = pc[mask]
        
        pc_image = np.zeros(image_size.astype(np.int32), dtype=np.float32)
        pc_image[grid_ids[:,1], grid_ids[:,0]] = pc[:, -1]
        # pc_image[pc_image==0] = pc[:, -1].mean()
        
        for i in range(pc_image.shape[0]):
            line = pc_image[i] 
            if line.mean() == 0: continue
            line[line==0] = line[line!=0].mean()
            pc_image[i] = line
        return np.expand_dims(pc_image, axis=0) 
    
    def _augmentation(self, input_dict):
        """random augmentation

        Args:
            input_dict (dict): raw input from getitem
            index (_type_): current index of the item

        Returns:
            input_dict: augmented input dict
        """
        
        input_dict = random_flip(input_dict, self.flip_mode)
        input_dict = random_crop(input_dict, self.crop_size)
        input_dict['depth_t1'] = self.pad_point_cloud(input_dict['depth_t1'])
        input_dict['depth_t2'] = self.pad_point_cloud(input_dict['depth_t2'])
        
        return input_dict
    
    def pad_point_cloud(self, pc, desired_len=16000):
        pc_len = pc.shape[0]
        # all padded points have coordinates -1 in xyz for masking
        paddings = -1 * np.ones((desired_len - pc_len, 3), dtype=np.float32)
        # pc_tensor = torch.from_numpy(pc).float()
        return np.concatenate([pc, paddings], axis=0)
    
    def unpad_point_cloud(self, pc):
        mask = pc[:,-1] != -1
        return pc[mask]
    
    def crop_point_cloud(self, pcd):
        sel_x = torch.logical_and(pcd[:,0]>self.bbox[0], pcd[:,0]<self.bbox[3])
        sel_y = torch.logical_and(pcd[:,1]>self.bbox[1], pcd[:,1]<self.bbox[4])
        sel_z = torch.logical_and(pcd[:,2]>self.bbox[2], pcd[:,2]<self.bbox[5])
        sel_xy = torch.logical_and(sel_x, sel_y)
        sel_xyz = torch.logical_and(sel_xy, sel_z)
        return pcd[sel_xyz]

class Full_DeFlow_Dataset(DeFlow_Dataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.depth_input_ratio = 0.9

    def __getitem__(self, index):
        next_idx = index + 1
        pcd1 = self._get_pointcloud(self.lidar_scans[index])
        pcd2 = self._get_pointcloud(self.lidar_scans[next_idx])
        # filter the point cloud to be only inside the view of camera 2
        pcd1 = self.LiCamT.FoV_filtering(pcd1)
        pcd2 = self.LiCamT.FoV_filtering(pcd2)
        
        # crop point cloud
        # pcd1, pcd2 = self.crop_point_cloud(pcd1), self.crop_point_cloud(pcd2)
        imgT_1 = self._get_image(self.imagesT[index])
        imgT_2= self._get_image(self.imagesT[next_idx])
        
        h_orig, w_orig = imgT_1.shape[-2:]
        input_im_size = np.array([h_orig, w_orig], dtype=np.float32)
        
        # convert point cloud to sparse depth on image
        # inversing depth here to make sure closer points would have a larger weights in l1_loss
        depth_t1 = self._get_inverse_depth(pcd1)
        depth_t2 = self._get_inverse_depth(pcd2)
        data = {
            'time_step': np.array([index, index+1]), 
            'imgT_1': imgT_1, # top-view camera T
            'imgT_2': imgT_2,
            'k_T': self.LiCamT.K,
            'rot_T': self.LiCamT.rot_mtx,
            'input_size': input_im_size,
            'depth_t1': depth_t1,
            'depth_t2': depth_t2,
            'flip': False,
            # 'min_z': min_z
            'tvec_T':self.LiCamT.t_vec,
            'k_S': self.LiCamS.K,
            'rot_S': self.LiCamS.rot_mtx,
            'tvec_S': self.LiCamT.t_vec
        }
        
        if self.input_depth == True:
            data = self._random_split_depth(data, self.depth_input_ratio)
        
        return data


class MultiFrame_DeFlow_Dataset(DeFlow_Dataset):
    def __init__(self, args, seq_length=5) -> None:
        self.args = args
        self.seq_length = seq_length
        
    def retrieve_one_frame():
        pass
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    
if __name__ == "__main__":
    
    test = DeFlow_Dataset('/scratch/liyzhu/debris_flow/DF_LiDAR_Data/')
    a = test[1]
    data_loader = DataLoader(
        DeFlow_Dataset('/scratch/liyzhu/debris_flow/DF_LiDAR_Data/'),
        batch_size=1, shuffle=False, drop_last=False, num_workers=12
    )
    from tqdm import tqdm
    l = []
    for data in tqdm(data_loader):
        l.append(data['depth_t1'].shape)
        
    pass