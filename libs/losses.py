from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.interpolation import my_grid_sample, get_coordgrid, interpolate2d_as
from models.deflow.modules_deflow import WarpingLayer_Flow


# code portion from https://github.com/MCG-NJU/CamLiFlow
def calc_census_loss_2d(image1, image2, noc_mask=None, max_distance=4):
    """
    Calculate photometric loss based on census transform.
    :param image1: [N, 3, H, W] float tensor
    :param image2: [N, 3, H, W] float tensor
    :param noc_mask: [N, 1, H, W] float tensor
    :param max_distance: int
    """
    def rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1) * 255.0

    def census_transform(gray_image):
        patch_size = 2 * max_distance + 1
        out_channels = patch_size * patch_size  # 9
        weights = torch.eye(out_channels, dtype=gray_image.dtype, device=gray_image.device)
        weights = weights.view([out_channels, 1, patch_size, patch_size])  # [9, 1, 3, 3]
        patches = nn.functional.conv2d(gray_image, weights, padding=max_distance)
        result = patches - gray_image
        result = result / torch.sqrt(0.81 + torch.pow(result, 2))
        return result

    if noc_mask is not None:
        image1 = noc_mask * image1
        image2 = noc_mask * image2

    gray_image1 = rgb_to_grayscale(image1)
    gray_image2 = rgb_to_grayscale(image2)

    t1 = census_transform(gray_image1)
    t2 = census_transform(gray_image2)

    dist = torch.pow(t1 - t2, 2)
    dist_norm = dist / (0.1 + dist)
    dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum

    n, _, h, w = image1.shape
    inner = torch.ones([n, 1, h - 2 * max_distance, w - 2 * max_distance], dtype=image1.dtype, device=image1.device)
    inner_mask = nn.functional.pad(inner, [max_distance] * 4)
    loss = dist_mean * inner_mask

    if noc_mask is not None:
        return loss.mean() / (noc_mask.mean() + 1e-7)
    else:
        return loss.mean()

@torch.cuda.amp.autocast(enabled=False)
def calc_smooth_loss_2d(image, flow, derivative='first'):
    """
    :param image: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param flow: [N, 2, H, W] float tensor
    :param derivative: 'first' or 'second'
    """
    def gradient(inputs):
        dy = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
        dx = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]
        return dx, dy

    image_dx, image_dy = gradient(image)
    flow_dx, flow_dy = gradient(flow)

    weights_x = torch.exp(-torch.mean(image_dx.abs(), 1, keepdim=True) * 10)
    weights_y = torch.exp(-torch.mean(image_dy.abs(), 1, keepdim=True) * 10)

    if derivative == 'first':
        loss_x = weights_x * flow_dx.abs() / 2.0
        loss_y = weights_y * flow_dy.abs() / 2.0
    elif derivative == 'second':
        flow_dx2 = flow_dx[:, :, :, 1:] - flow_dx[:, :, :, :-1]
        flow_dy2 = flow_dy[:, :, 1:, :] - flow_dy[:, :, :-1, :]
        loss_x = weights_x[:, :, :, 1:] * flow_dx2.abs()
        loss_y = weights_y[:, :, 1:, :] * flow_dy2.abs()
    else:
        raise NotImplementedError('Unknown derivative: %s' % derivative)

    return loss_x.mean() / 2 + loss_y.mean() / 2


def calc_ssim_loss_2d(image1, image2, noc_mask=None, max_distance=4):
    """
    Calculate photometric loss based on SSIM.
    :param image1: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param image2: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param noc_mask: [N, 1, H, W] float tensor, ranging from 0 to 1
    :param max_distance: int
    """
    patch_size = 2 * max_distance + 1
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    if noc_mask is not None:
        image1 = noc_mask * image1
        image2 = noc_mask * image2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(image1)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(image2)
    mu_x_square, mu_y_square = mu_x.pow(2), mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(image1 * image1) - mu_x_square
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(image2 * image2) - mu_y_square
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(image1 * image2) - mu_xy

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x_square + mu_y_square + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    loss = torch.clamp((1 - ssim) / 2, min=0.0, max=1.0)

    if noc_mask is not None:
        return loss.mean() / (noc_mask.mean() + 1e-7)
    else:
        return loss.mean()

def compute_depth_error_metrics(gt, pred, mask):
    """
    Compute the absolute error(L1) of depth prediction
    The gt and prediction are expressed in the reciprocal of depth
    Args:
        gt (_type_): _description_
        pred (_type_): _description_
        mask: invalide values
    """
    gt_depth = 5.0 / gt[...,-1].squeeze().detach()
    pred_depth = 5.0 / pred.squeeze().detach()
    
    # compute error within 10m, 30m, 50m
    mask10 = (pred_depth <= 10.0) * mask
    mask30 = (pred_depth <= 30.0) * mask
    mask50 = (pred_depth <= 50.0) * mask
    depth_error = torch.abs(gt_depth - pred_depth) 
    abs_err10 = (depth_error * mask10).sum() / (mask10.sum() + 1.0e-5)
    abs_err30 = (depth_error * mask30).sum() / (mask30.sum() + 1.0e-5)
    abs_err50 = (depth_error * mask50).sum() / (mask50.sum() + 1.0e-5)
    
    # compute relative errors
    rel_error = (depth_error / gt_depth * mask30).sum()  / (mask30.sum() + 1.0e-5)

    return abs_err10, abs_err30, abs_err50, rel_error

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, pred, gt, mask):
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(gt)-torch.log(pred))[mask] ** 2 ))
        return loss
    
class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
    
    def forward(self, pred, gt, mask):
        loss = torch.abs(gt - pred)[mask].mean()
        return loss

class SceneFlow_Loss(nn.Module):
    """Loss function design for debris flow estimation
    Semi-supervision
    + Direct depth supervision from LiDAR
    + Optical Flow Loss self-supervised

    Args:
        nn (_type_): _description_
    """
    def __init__(self, args) -> None:
        super(SceneFlow_Loss, self).__init__()
        
        self._weights = args.loss.pyramid_weights
        self._flow_photo_w = args.loss.flow_photo_w
        self._depth_l1_w = args.loss.depth_l1_w
        self._backwarp_flow = WarpingLayer_Flow()
        self.smooth_derivative = args.loss.smooth_derivative
        self.photometric_loss = args.loss.photometric_loss
        self.compute_depth = args.train.compute_depth
        self._depth_w = args.loss.depth_weight
        self._flow_w = args.loss.flow_weight
        self.search_range = args.loss.photo_max_d
        
        if args.loss.depth_data_loss == 'L1':
            self.data_data_loss = L1_loss()
        elif args.loss.depth_data_loss == 'RMSELog': 
            self.data_data_loss = RMSE_log()
        else: raise NotImplementedError
        
    def detect_occlusion(self, flow_f, flow_b):
        flow_b_warped = self._backwarp_flow(flow_b, flow_f)
        flow_f_warped = self._backwarp_flow(flow_f, flow_b)
        
        diff_flow_f = (flow_f + flow_b_warped)
        diff_flow_b = (flow_b + flow_f_warped)
        
        diff_mag_f = diff_flow_f[:,0]**2 + diff_flow_f[:,1]**2
        return 
    
    def depth_prior_loss(self, input_dict, output_dict):
        
        k = input_dict['k_T']
        R = input_dict['rot_T']
        t = input_dict['tsfm_T']
        depth_prior_loss = 0
        min_z = input_dict['min_z']
        from kornia.geometry.camera.perspective import unproject_points
        for (depth1, depth2) in zip(output_dict['depth_t1'], output_dict['depth_t2']):
            depth1_scaled = interpolate2d_as(depth1, input_dict['imgT_1'])
            depth2_scaled = interpolate2d_as(depth2, input_dict['imgT_1'])
            pts_grid = get_coordgrid(depth1_scaled).permute(0,2,3,1)[...,:2]
            pc1 = unproject_points(pts_grid, 5.0/depth1_scaled.permute(0,2,3,1), k)
            pc2 = unproject_points(pts_grid, 5.0/depth2_scaled.permute(0,2,3,1), k)
            pc1_world = (R.transpose(1, 2) @ pc1.permute(0,1,3,2) - R.transpose(1, 2) @ t).permute(0,1,3,2)
            pc2_world = (R.transpose(1, 2) @ pc2.permute(0,1,3,2) - R.transpose(1, 2) @ t).permute(0,1,3,2)
            z1 = pc1_world[...,-1]
            z2 = pc2_world[...,-1]
            mask1 = z1 < min_z
            mask2 = z2 < min_z
            depth_prior_loss += ((min_z-z1) * mask1).mean()
            depth_prior_loss += ((min_z-z2) * mask2).mean()
            
        return depth_prior_loss
    
    def depth_loss(self, input_dict, output_dict):
        # b = output_dict['depth_t1'][0].shape[0]
        # compute the dif between pred depth and lidar data
        gt1 = input_dict['depth_t1']
        gt2 = input_dict['depth_t2']
        # input depth is padded with invalid values, retrieve the valid mask
        gt1_mask = gt1[:,:,-1] != -1
        gt2_mask = gt2[:,:,-1] != -1
        
        grid1 = gt1[:,:,:2].float()
        grid2 = gt2[:,:,:2].float()
        
        h, w = input_dict['input_size'][0]
        
        grid1[:,:,0] = grid1[:,:,0] / w * 2 - 1 # x
        grid1[:,:,1] = grid1[:,:,1] / h * 2 - 1 # y
        grid1 = grid1.unsqueeze(1)
        
        grid2[:,:,0] = grid2[:,:,0] / w * 2 - 1 # x
        grid2[:,:,1] = grid2[:,:,1] / h * 2 - 1 # y
        grid2 = grid2.unsqueeze(1)
        
        depth_l1_loss = 0
        # abs_e10, abs_e30, abs_e50 = 0, 0, 0
        for i, (pred1, pred2) in enumerate(zip(output_dict['depth_t1'], output_dict['depth_t2'])):
            # depth 1
            sampled_pred = my_grid_sample(pred1, grid1).squeeze()
            depth_l1 = self.data_data_loss(sampled_pred, gt1[:,:,-1], gt1_mask)
            abs_es_1 = compute_depth_error_metrics(gt1, sampled_pred, gt1_mask)
            
            # depth 2
            sampled_pred = my_grid_sample(pred2, grid2).squeeze()
            depth_l2 = self.data_data_loss(sampled_pred, gt2[:,:,-1], gt2_mask)
            abs_es_2 = compute_depth_error_metrics(gt2, sampled_pred, gt2_mask)
            
            # abs_err /= 2.0
            depth_l1_loss += depth_l1 + depth_l2
            depth_errors = [(e1+e2)/2.0 for (e1,e2) in zip(abs_es_1, abs_es_2)]
        
        # smoothness loss 
        smooth_loss = calc_smooth_loss_2d(input_dict['imgT_1'], 1/output_dict['depth_t1'][0], derivative=self.smooth_derivative)
        smooth_loss += calc_smooth_loss_2d(input_dict['imgT_2'], 1/output_dict['depth_t2'][0], derivative=self.smooth_derivative)
        # depth_prior = self.depth_prior_loss(input_dict, output_dict)
        # smooth_loss *= 10
        depth_total_loss = depth_l1_loss * self._depth_l1_w + (1.0 - self._depth_l1_w) * smooth_loss
        
        
        return {'depth_total_loss': depth_total_loss,
                'depth_smooth_loss': smooth_loss,
                'depth_l1_loss': depth_l1_loss,
                # 'depth_abs_err': abs_err,
                'depth_err10': depth_errors[0],
                'depth_err30': depth_errors[1],
                'depth_err50': depth_errors[2],
                # 'depth_prior': depth_prior
                'rel_depth_err': depth_errors[3]
                }
    
    def flow_loss(self, input_dict, output_dict):
        photo_loss = 0
        smooth_loss = 0
        image1 = input_dict['imgT_1']
        image2 = input_dict['imgT_2']
        for lv, (flow_f, flow_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'])):
            curr_hw = (flow_f.shape[2], flow_f.shape[3])
            image1_scaled = F.interpolate(image1, curr_hw, mode='area')
            image2_scaled = F.interpolate(image2, curr_hw, mode='area')
            
            image1_scaled_warp = self._backwarp_flow(image1_scaled, flow_b)
            image2_scaled_warp = self._backwarp_flow(image2_scaled, flow_f)
            
            # calculate photometric loss
            if self.photometric_loss == 'ssim':
                photo_loss1 = calc_ssim_loss_2d(image1_scaled, image2_scaled_warp, max_distance=self.search_range)
                photo_loss2 = calc_ssim_loss_2d(image2_scaled, image1_scaled_warp, max_distance=self.search_range)
            elif self.photometric_loss == 'census':
                photo_loss1 = calc_census_loss_2d(image1_scaled, image2_scaled_warp, max_distance=self.search_range)
                photo_loss2 = calc_census_loss_2d(image2_scaled, image1_scaled_warp, max_distance=self.search_range)
            else:
                raise NotImplementedError(f'Unknown photometric loss: {self.photometric_loss}')
            
            photo_loss += self._weights[lv] * (photo_loss1 + photo_loss2) / 2
            
            # calculate smooth loss
            scale = min(flow_f.shape[2], flow_b.shape[3])
            smooth_loss1 = calc_smooth_loss_2d(image1_scaled, flow_f / scale, self.smooth_derivative)
            smooth_loss2 = calc_smooth_loss_2d(image2_scaled, flow_b / scale, self.smooth_derivative)
            smooth_loss += self._weights[lv] * (smooth_loss1 + smooth_loss2) / 2
            # print(f'{lv}:{photo_loss}, {smooth_loss}')
            # self.detect_occlusion(flow_f, flow_b)
        
        flow_loss = photo_loss * self._flow_photo_w + (1 - self._flow_photo_w) * smooth_loss
            
        return {'flow_loss': flow_loss,
                'flow_photo_loss': photo_loss,
                'flow_smooth_loss': smooth_loss
                }
    
    def static_loss(self, output_dict):
        flow_b = output_dict['flow_b'][0]
        flow_f = output_dict['flow_f'][0]
        depth1 = output_dict['depth_t1'][0]
        depth2 = output_dict['depth_t2'][0]
        static_mask = output_dict['static_mask']
        # flow static loss
        # static_flow_loss = (flow_b.abs() * static_mask).sum() / static_mask.sum()
        # static_flow_loss += (flow_f.abs() * static_mask).sum() / static_mask.sum()
        # depth static loss
        static_depth_loss = ((depth1 - depth2).abs() * static_mask).mean()
        # print((static_depth_loss + static_flow_loss).item())
        return static_depth_loss
    
    def cycle_loss(self, output_dict):
        cycle_loss = output_dict['forward_occ'].mean()
        cycle_loss += output_dict['backward_occ'].mean()
        return cycle_loss
    
    def forward(self, input_dict, output_dict):
        loss_dict = dict()
        # if self.compute_depth == True:
        flow_loss_dict = self.flow_loss(input_dict, output_dict)
        depth_loss_dict = self.depth_loss(input_dict, output_dict)
        static_loss = self.static_loss(output_dict)
        cycle_loss = self.cycle_loss(output_dict)
        # static_loss = 0
        
        total_loss = flow_loss_dict['flow_loss'] * self._flow_w + depth_loss_dict['depth_total_loss'] * self._depth_w + static_loss + cycle_loss
            
        # elif self.compute_depth == False:
        #     flow_loss_dict = self.flow_loss(input_dict, output_dict)
        #     depth_loss_dict = dict()
        #     loss_dict['depth_total_loss'] = torch.zeros(1).cuda()
            
        #     total_loss = flow_loss_dict['flow_loss']
        
        loss_dict['static_loss'] = static_loss
        loss_dict['cycle_loss'] = cycle_loss
        loss_dict['total_loss'] = total_loss
        loss_dict.update(flow_loss_dict)
        loss_dict.update(depth_loss_dict)
        
        # print(loss_dict['flow_loss'].item(), total_loss.item())
        return loss_dict


class RMSE_2d(nn.Module):
    def __init__(self):
        super(RMSE_2d, self).__init__()
    
    def forward(self, img1, img2):
        img1 = img1 * 128 + 128
        img2 = img2 * 128 + 128
        return ((img1 - img2) ** 2).sum(1).mean().sqrt()
    
