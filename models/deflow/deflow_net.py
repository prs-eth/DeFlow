## code partially adpated from 
## https://github.com/visinf/multi-mono-sf
## Liyuan Zhu
from __future__ import absolute_import, division
from matplotlib.pyplot import axis

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_deflow import *


class DeFlowNet(nn.Module):
    def __init__(self, args):
        super(DeFlowNet, self).__init__()
        self.args = args
        
        # 6 layer feature pyramid from multi-mono-sf and uflow
        self.img_chs = self.args.network.num_channels
        self.depth_chs = self.args.network.depth_channels
        self.feature_extractor = FeatureExtractor(self.img_chs)
        self.depth_extractor = FeatureExtractor(self.depth_chs)
        self.warping_layer_flow = WarpingLayer_Flow()
        
        self.correlation = compute_cost_volume
        
        self.search_range = self.args.network.search_range
        self.output_level = self.args.network.output_level
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=False)
        
        self.corr_params = {
            "pad_size": 0,
            "kernel_size": 1, 
            "max_disp": self.search_range, 
            "stride1": 1, 
            "stride2": 1, 
            "corr_multiply": 1
        }
        
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for l, (img_ch, d_ch) in enumerate(zip(self.img_chs[::-1], self.depth_chs[::-1])):
            if l > self.output_level:
                break
            if l == 0:
                num_ch_in = self.dim_corr + img_ch + d_ch
            else:
                num_ch_in = self.dim_corr + img_ch + d_ch + 96 + 2 + 1 # 96->out_x, 2->flow, 1->dep
                self.upconv_layers.append(upconv(96, 96, 3, 2))
            
            sf_layer = DeFlowDecoder(num_ch_in)
            self.flow_estimators.append(sf_layer)
            
        self.sigmoid = torch.nn.Sigmoid()
        self.context_networks = ContextNetwork(96 + 2 + 1)
        
        initialize_msra(self.modules())
        
    def pwc_forward(self, input_dict):
        """ Enhanced pwd network from uflow and multi-mono-sf
            + use 6 layer instead of 7
            + cost volume normalization
            + remove context network
            + splitted decoder design
        Args:
            input_dict (_type_): _description_
        """

        output_dict = {}
        
        # generate feature pyramid for 2 frames and depths
        x1, x2 = input_dict['imgT_1'], input_dict['imgT_2']
        d1, d2 = input_dict['input_depth_t1'], input_dict['input_depth_t2']
        fp_x1 = self.feature_extractor(x1) + [x1]
        fp_x2 = self.feature_extractor(x2) + [x2]
        fp_d1 = self.depth_extractor(d1) + [d1]
        fp_d2 = self.depth_extractor(d2) + [d2]
        
        # outputs
        flows_f = []
        flows_b = []
        depths_1 = []
        depths_2 = []
        
        # warp the feature map according to flow estimate
        for l, (x1, x2, d1, d2) in enumerate(zip(fp_x1, fp_x2, fp_d1, fp_d2)):
            if l == 0:
                x1_warp = x1
                x2_warp = x2
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                depth_t1 = interpolate2d_as(depth_t1, x1, mode="bilinear")
                depth_t2 = interpolate2d_as(depth_t2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                # k1, k2 = input_dict['k_top'].float(), input_dict['k_top'].float()
                x2_warp = self.warping_layer_flow(x2, flow_f)
                x1_warp = self.warping_layer_flow(x1, flow_b)
    
            # compute correlation and activation
            # normalization gives better results
            x1, x2, x1_warp, x2_warp = normalize_features([x1, x2, x1_warp, x2_warp])
            corr_f = self.correlation(x1, x2_warp, self.corr_params)
            corr_b = self.correlation(x2, x1_warp, self.corr_params)
            corr_f_relu, corr_b_relu = self.leakyRELU(corr_f), self.leakyRELU(corr_b)
            
            # estimator
            if l == 0:
                x1_out, flow_f, depth_t1 = self.flow_estimators[l](torch.cat([corr_f_relu, x1, d1], dim=1))
                x2_out, flow_b, depth_t2 = self.flow_estimators[l](torch.cat([corr_b_relu, x2, d2], dim=1))
            else:
                x1_out, flow_f_res, depth_t1 = self.flow_estimators[l](torch.cat([corr_f_relu, x1, x1_out, flow_f, depth_t1, d1], dim=1)) #num_ch_in
                x2_out, flow_b_res, depth_t2 = self.flow_estimators[l](torch.cat([corr_b_relu, x2, x2_out, flow_b, depth_t2, d2], dim=1)) #num_ch_in
                flow_f += flow_f_res
                flow_b += flow_b_res
            
            if l!= self.output_level:
                depth_t1 = self.sigmoid(depth_t1)
                depth_t2 = self.sigmoid(depth_t2)
                flows_b.append(flow_b)
                flows_f.append(flow_f)
                depths_1.append(depth_t1)
                depths_2.append(depth_t2)
            else:
                flow_res_f, depth_t1 = self.context_networks(torch.cat([x1_out, flow_f, depth_t1], dim=1))
                flow_res_b, depth_t2 = self.context_networks(torch.cat([x2_out, flow_b, depth_t2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                flows_f.append(flow_f)
                flows_b.append(flow_b)
                depths_1.append(depth_t1)
                depths_2.append(depth_t2) 
                break 
            # if l == self.output_level: break

        x1_rev = fp_x1[::-1]
        
        output_dict['flow_f'] = upsample_outputs_as(flows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(flows_b[::-1], x1_rev)
        output_dict['depth_t1'] = upsample_outputs_as(depths_1[::-1], x1_rev)
        output_dict['depth_t2'] = upsample_outputs_as(depths_2[::-1], x1_rev)
        
        return output_dict
    
    def post_precessing(self, output_dict):
        # compute static mask
        flow_f = output_dict['flow_f'][0]
        flow_b = output_dict['flow_b'][0]
        flow_mag = compute_flow_magnitude(flow_b) + compute_flow_magnitude(flow_f)
        # flow_mean = torch.mean(flow_mag, dim=(1,2,3), keepdim=True)
        static_mask = flow_mag <= 2.0
        output_dict['static_mask'] = static_mask
        
        # check forward backward consistency
        fwd_occ, bwd_occ = forward_backward_consistency_check(flow_f, flow_b)
        output_dict['forward_occ'] = fwd_occ
        output_dict['backward_occ'] = bwd_occ
        
        return output_dict
    
    def forward(self, input_dict):
        
        output_dict = self.pwc_forward(input_dict)
        
        return self.post_precessing(output_dict)
            