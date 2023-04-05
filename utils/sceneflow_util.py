from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from utils.interpolation import interpolate2d, my_grid_sample, get_coordgrid, get_grid
import numpy as np
from matplotlib.colors import hsv_to_rgb
from itertools import accumulate
from collections import namedtuple
from typing import Optional
import open3d as o3d
DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

BLUE = (94/255, 129/255, 160/255)
GREEN = (163/255, 190/255, 128/255)
RED = (191/255, 97/255, 106/255)
PURPLE = (180/255, 142/255, 160/255)
OPACITY = 1.0

def post_processing(l_disp, r_disp):
    
    b, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    grid_l = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w).to(device=l_disp.device, dtype=l_disp.dtype).requires_grad_(False)
    l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), 0, 1)
    r_mask = torch.flip(l_mask, [3])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def flow_horizontal_flip(flow_input):

    flow_flip = torch.flip(flow_input, [3])
    flow_flip[:, 0:1, :, :] *= -1

    return flow_flip.contiguous()


def disp2depth_kitti(pred_disp, k_value, depth_clamp=True):

    pred_depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-4)
    if depth_clamp:
        pred_depth = torch.clamp(pred_depth, 1e-3, 80)

    return pred_depth


def depth2disp_kitti(pred_depth, k_value, depth_clamp=True):

    if depth_clamp:
        pred_depth = torch.clamp(pred_depth, 1e-3, 80)
    pred_disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_depth)

    return pred_disp


def pixel2pts(intrinsics, depth):
    b, _, h, w = depth.size()

    pixelgrid = get_coordgrid(depth)

    depth_mat = depth.view(b, 1, -1)
    pixel_mat = pixelgrid.view(b, 3, -1)
    # doing torch.inverse on CPU is way faster than that on GPU
    pts_mat = torch.matmul(torch.inverse(intrinsics.float().cpu()).to(device=depth.device, dtype=depth.dtype), pixel_mat) * depth_mat
    pts = pts_mat.view(b, -1, h, w)

    return pts, pixelgrid


def pts2pixel(pts, intrinsics):

    b, _, h, w = pts.size()

    proj_pts = torch.matmul(intrinsics, pts.view(b, 3, -1))
    pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]
    pixels_mat = torch.clamp(pixels_mat, -w * 1.5, w * 1.5)

    return pixels_mat.view(b, 2, h, w)


def intrinsic_scale(intrinsic, scale_y, scale_x):
    b, h, w = intrinsic.size()
    fx = intrinsic[:, 0, 0] * scale_x
    fy = intrinsic[:, 1, 1] * scale_y
    cx = intrinsic[:, 0, 2] * scale_x
    cy = intrinsic[:, 1, 2] * scale_y

    zeros = torch.zeros_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim=1)
    r2 = torch.stack([zeros, fy, cy], dim=1)
    r3 = torch.tensor([0., 0., 1.], device=intrinsic.device, dtype=intrinsic.dtype, requires_grad=False).unsqueeze(0).expand(b, -1)
    intrinsic_s = torch.stack([r1, r2, r3], dim=1)

    return intrinsic_s


def pixel2pts_ms(intrinsic, output_disp, rel_scale, depth_clamp=True):
    # pixel2pts
    intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
    output_depth = disp2depth_kitti(output_disp, intrinsic_dp_s[:, 0, 0], depth_clamp)
    pts, _ = pixel2pts(intrinsic_dp_s, output_depth)

    return pts, intrinsic_dp_s


def pts2pixel_ms(intrinsic, pts, output_sf, disp_size):

    # +sceneflow and reprojection
    sf_s = interpolate2d(output_sf, disp_size, mode="bilinear")
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)

    norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)

    return sf_s, pts_tform, norm_coord


def reconstructImg(coord, img):
    grid = coord.transpose(1, 2).transpose(2, 3)
    img_warp = my_grid_sample(img, grid)

    mask = torch.ones_like(img, requires_grad=False)
    mask = my_grid_sample(mask, grid)
    mask = (mask >= 1.0).to(dtype=img.dtype)
    return img_warp * mask


def reconstructPts(coord, pts):
    grid = coord.transpose(1, 2).transpose(2, 3)
    pts_warp = my_grid_sample(pts, grid)

    mask = torch.ones_like(pts, requires_grad=False)
    mask = my_grid_sample(mask, grid)
    mask = (mask >= 1.0).to(dtype=pts.dtype)
    return pts_warp * mask


def projectSceneFlow2Flow_kitti(intrinsic, sceneflow, disp, input_size=None, depth_clamp=True):

    _, _, h, w = disp.size()

    if input_size == None:
        output_depth = disp2depth_kitti(disp, intrinsic[:, 0, 0], depth_clamp)
        pts, pixelgrid = pixel2pts(intrinsic, output_depth)
    else:               ## if intrinsic is not adjusted to the "input_size"
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h
        local_scale[:, 1] = w
        pts, intrinsic = pixel2pts_ms(intrinsic, disp, local_scale / input_size)
        pixelgrid = get_coordgrid(disp)

    sf_s = interpolate2d(sceneflow, [h, w], mode="bilinear")
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    flow = coord - pixelgrid[:, 0:2, :, :]

    return flow

def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()

    return pixelgrid

def make_colorwheel(transitions: tuple=DEFAULT_TRANSITIONS) -> np.ndarray:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array, ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0])
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index
    return colorwheel


def sceneflow_to_rgb(
        flow: np.ndarray,
        flow_max_radius: Optional[float]=None,
        background: Optional[str]="bright",
    ) -> np.ndarray:
        """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
        Args:
            flow: scene flow.
                flow[..., 0] should be the x-displacement
                flow[..., 1] should be the y-displacement
                flow[..., 2] should be the z-displacement
            flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
                Default: The normalization is based on the input flow maximum radius.
            background: States if zero-valued flow should look 'bright' or 'dark'.
        Returns: An array of RGB colors.
        """
        valid_backgrounds = ("bright", "dark")
        if background not in valid_backgrounds:
            raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}.")
        wheel = make_colorwheel()
        # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
        complex_flow = flow[..., 0] + 1j * flow[..., 1]
        radius, angle = np.abs(complex_flow), np.angle(complex_flow)
        if flow_max_radius is None:
            flow_max_radius = np.max(radius)
        if flow_max_radius > 0:
            radius /= flow_max_radius
        ncols = len(wheel)
        # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
        angle[angle < 0] += 2 * np.pi
        angle = angle * ((ncols - 1) / (2 * np.pi))
        # Make the wheel cyclic for interpolation
        wheel = np.vstack((wheel, wheel[0]))
        # Interpolate the hues
        (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
        angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
        float_hue = (
            wheel[angle_floor.astype(np.int)] * (1 - angle_fractional) + wheel[angle_ceil.astype(np.int)] * angle_fractional
        )
        ColorizationArgs = namedtuple(
            'ColorizationArgs', ['move_hue_valid_radius', 'move_hue_oversized_radius', 'invalid_color']
        )
        def move_hue_on_V_axis(hues, factors):
            return hues * np.expand_dims(factors, -1)
        def move_hue_on_S_axis(hues, factors):
            return 255. - np.expand_dims(factors, -1) * (255. - hues)
        if background == "dark":
            parameters = ColorizationArgs(
                move_hue_on_V_axis, move_hue_on_S_axis, np.array([255, 255, 255], dtype=np.float)
            )
        else:
            parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, np.array([0, 0, 0], dtype=np.float))
        colors = parameters.move_hue_valid_radius(float_hue, radius)
        oversized_radius_mask = radius > 1
        colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
            float_hue[oversized_radius_mask],
            1 / radius[oversized_radius_mask]
        )
        return colors.astype(np.uint8)


def tensor_flow2rgb(flow, max_value=256):
    n = 8
    np_flow = flow.detach().cpu().numpy()
    u, v = np_flow[0], np_flow[1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    
    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_value, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)
    
    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)
    return image_rgb.transpose(2,0,1)

def warp_flow(flow, x):
    flo_list = []
    flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1) #normalize flow to [-1, 1]
    flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
    flo_list.append(flo_w)
    flo_list.append(flo_h)
    flow_for_grid = torch.stack(flo_list).transpose(0, 1)
    grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
    x_warp = my_grid_sample(x, grid)

    # mask out the flow outside
    mask = torch.ones_like(x, requires_grad=False)
    mask = my_grid_sample(mask, grid)
    mask = (mask > 0.999).to(dtype=x_warp.dtype)

    return x_warp * mask

def point_cloud_from_rgbd(rgb, depth, k):
    
    # all numpy matrices
    rgb = np.transpose(rgb, (1,0,2))
    rgb = np.asarray(rgb/255.0, dtype=np.float64)
    w, h = depth.shape
    grid_x = np.expand_dims(np.linspace(0.0, w-1, w), axis=1) 
    grid_y = np.expand_dims(np.linspace(0.0, h-1, h), axis=0) 
    ones = np.ones_like(grid_y)
    # coordgrid = np.concatenate((grid_x, grid_y, ones))
    xv, yv = np.meshgrid(grid_x, grid_y)
    
    # grid 2d [3, h, w] (u,v,1)
    grid_2d = np.stack((xv, yv, depth.T))
    grid_2d[0] = grid_2d[0] * grid_2d[2]
    grid_2d[1] = grid_2d[1] * grid_2d[2]
    xyz = np.linalg.inv(k) @ np.reshape(grid_2d, (3, -1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)
    pcd.colors = o3d.utility.Vector3dVector(np.reshape(rgb, (-1, 3)))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 5], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def sceneflow_from_2d(flow, depth1, depth2, k):
    """
    compute 3d sceneflow from 2d depth and optical flow

    Args:
        flow (tensor): _description_
        depth1 (tensor): _description_
        depth2 (tensor): _description_
        k (tensor): _description_
    """
    b,_,h,w = depth1.shape
    mag_flow = (flow ** 2).sum(1, keepdim=True)
    # static_mask = (mag_flow < mag_flow.mean() / 5).squeeze().detach().cpu().numpy()
    grid_2d = get_coordgrid(depth1)
    depth1 = (depth1 + depth2) / 2
    depth2 = (depth1 + depth2) / 2
    depth1 = 5.0/depth1
    depth2 = 5.0/depth2
    # depth2_warped = warp_flow(flow, depth2)
    grid_1 = grid_2d * depth1
    grid_2 = grid_2d * depth2
    pts_3d_1 = k.inverse() @ grid_1.view(b,3,-1)
    pts_3d_2 = k.inverse() @ grid_2.view(b,3,-1)
    pts_3d_1 = pts_3d_1.view(b,3,h,w)
    pts_3d_2 = pts_3d_2.view(b,3,h,w)
    pts_3d_2_warped = warp_flow(flow, pts_3d_2)
    sceneflow = (pts_3d_2_warped - pts_3d_1).squeeze().permute(1,2,0)
    
    sceneflow = sceneflow.detach().cpu().numpy()
    # sceneflow *= np.repeat(np.expand_dims(static_mask, 3), 3, axis=3)
    sf_mag = np.sqrt((sceneflow ** 2).sum(-1))
    static_mask = mag_flow >= 0.5
    # sf_mag[static_mask] = 0
    static_mask = static_mask.squeeze().detach().cpu().numpy()
    from scipy.ndimage.morphology import binary_dilation
    st_mask = binary_dilation(static_mask)
    sceneflow[st_mask!=True] = 0
    return sf_mag, pts_3d_1.squeeze().detach().cpu().numpy()
    sf_rgb = sceneflow_to_rgb(sceneflow, flow_max_radius=5)
    # sf_rgb = np.transpose(sf_rgb, axes=(0,3,1,2))
    import matplotlib.pyplot as plt
    mag_mask = (sf_mag<0.5) * (sf_mag>0.05)
    plt.hist(sf_mag[static_mask * mag_mask], bins='auto')
    plt.show()
    # import torchshow as ts
    # ts.show(sf_rgb)
    
    # pts_np = pts_3d_1.squeeze().detach().cpu().numpy()
    # pts_np = np.transpose(pts_np, (1, 2, 0))
    # sf_mag = np.reshape(sf_mag, (-1, 1))
    # pcd_color = np.reshape(sf_rgb, (3,-1)).T
    
    # depth = np.asarray(depth, order='C')
    # # rgbd = torch.cat([input_dict['imgT_1'], output_dict['depth_t1'][0]], dim=1).squeeze().permute(1,2,0).detach().cpu().numpy()

    # k = k.squeeze().detach().cpu().numpy().astype(np.float64)
    # pcd = point_cloud_from_rgbd(rgb, 5.0/depth, k)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts_np)
    # pcd.colors = o3d.utility.Vector3dVector(pcd_color/255.0)
    # vis = o3d.visualization.Visualizer()
    
    # vis.create_window()
    
    # vis.add_geometry(pcd)
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # o3d.visualization.draw_geometries([pcd])
    return sf_rgb, st_mask

class bounding_box_2d():
    def __init__(self, w, h, x, y):
        self.w = w
        self.h = h
        self.x = x
        self.y = y
    
    def _crop_image(self, image):
        return image[self.y:self.y+self.h, self.x:self.x+self.w]
    
class bounding_box_xy():
    """ 2.5d bounding box for camera-LiDAR system
    
    initialized with min, max for x and y coordinates
    """
    def __init__(self, CamLi, x0=-1.0, x1=1.0, y0=19.0, y1=21.0):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.CamLi = CamLi
    
    def _crop_3d(self, coord, flow):
        h, w = coord.shape[1:]
        coord = np.reshape(coord, (3, h * w))
        coord_w = self.CamLi.T_cam2world(coord)
        coord_w = np.reshape(coord_w, (3, h, w))
        mask_x = np.logical_and(coord_w[0,:]>self.x0, coord_w[0,:]<self.x1)
        mask_y = np.logical_and(coord_w[1,:]>self.y0, coord_w[1,:]<self.y1)
        mask = np.logical_and(mask_x, mask_y)
        return flow[mask]
    
    