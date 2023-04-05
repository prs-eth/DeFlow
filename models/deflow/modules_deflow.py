from __future__ import absolute_import, division
from numpy import disp

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 

from utils.interpolation import interpolate2d_as, my_grid_sample, get_grid
# from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, pixel2pts, pts2pixel


def merge_lists(input_list):
    """
    input_list = list:time[ list:level[4D tensor] ]
    output_list = list:level[ 4D tensor (batch*time, channel, height, width)]
    """
    len_tt = len(input_list)
    len_ll = len(input_list[0])

    output_list = []
    for ll in range(len_ll):
        list_ll = []
        for tt in range(len_tt):
            list_ll.append(input_list[tt][ll])            
        tensor_ll = torch.stack(list_ll, dim=1)
        tbb, ttt, tcc, thh, tww = tensor_ll.size()
        output_list.append(tensor_ll.reshape(tbb * ttt, tcc, thh, tww))

    return output_list


# https://github.com/google-research/google-research/blob/789d828d545dc35df8779ad4f9e9325fc2e3ceb0/uflow/uflow_model.py#L88
def compute_cost_volume(feat1, feat2, param_dict):
    """
    only implemented for:
        kernel_size = 1
        stride1 = 1
        stride2 = 1
    """

    max_disp = param_dict["max_disp"]

    _, _, height, width = feat1.size()
    num_shifts = 2 * max_disp + 1
    feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], axis=1, keepdims=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, axis=1)
    return cost_volume


# https://github.com/google-research/google-research/blob/789d828d545dc35df8779ad4f9e9325fc2e3ceb0/uflow/uflow_model.py#L44
def normalize_features(feature_list):

    statistics_mean = []
    statistics_var = []
    axes = [-3, -2, -1]

    for feature_image in feature_list:
        statistics_mean.append(feature_image.mean(dim=axes, keepdims=True))
        statistics_var.append(feature_image.var(dim=axes, keepdims=True))

    statistics_std = [torch.sqrt(v + 1e-16) for v in statistics_var]

    feature_list = [f - mean for f, mean in zip(feature_list, statistics_mean)]
    feature_list = [f / std for f, std in zip(feature_list, statistics_std)]

    return feature_list


class WarpingLayer_Flow(nn.Module):
    """
    Backward warp an input tensor "x" using the input optical "flow"
    """
    def __init__(self):
        super(WarpingLayer_Flow, self).__init__()

    def forward(self, x, flow):
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



def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample_outputs_as(input_list, ref_list):
    """
    upsample the tensor in the "input_list" with a size of tensor included in "ref_list"

    """
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

    return output_list


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, padding_mode="zeros"):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size=kernel_size, padding_mode=padding_mode)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs, padding_mode="zeros"):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2, padding_mode=padding_mode),
                conv(ch_out, ch_out, padding_mode=padding_mode)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv_ii in self.convs:
            x = conv_ii(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]

class DeFlowDecoder(nn.Module):
    """ Splitted decoder design from  https://github.com/visinf/multi-mono-sf
        I follow the results from the multi-mono-SF
        remove the context network and use the splitting at 2nd-to-last layer decoder
        Liyuan Zhu
    """
    def __init__(self, ch_in):
        super(DeFlowDecoder, self).__init__()
        
        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96)
        )
        self.conv_flow = nn.Sequential(
            conv(96, 64),
            conv(64, 32),
            conv(32, 2, isReLU=False)
        )
        self.conv_d1 = nn.Sequential(
            conv(96, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
        )
        
    def forward(self, x):
        x_curr = self.convs(x)
        flow = self.conv_flow(x_curr)
        depth = self.conv_d1(x_curr)
        
        return x_curr, flow, depth


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        # dilated convolutional layer to enlarge the perceptive field
        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1)
        )
        self.conv_flow = conv(32, 2, isReLU=False)
        self.conv_depth = nn.Sequential(
            conv(32, 1, isReLU=False), 
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_flow(x_out)
        disp1 = self.conv_depth(x_out)

        return sf, disp1
    
def compute_flow_magnitude(flow, dim=1):
    """Compute the flow magnitude

    Args:
        flow (tensor):  
        dim (int, optional): the dimension of the flow. Defaults to 1.
    Author: Liyuan Zhu
    """
    mag = torch.sqrt((flow ** 2).sum(dim, keepdim=True))
    return mag


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_warp = WarpingLayer_Flow()
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ