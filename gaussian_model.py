import os 
import sys 
import torch 
import torch.nn as nn 
import numpy as np 
from utils.general_utils import * 
from utils.sh_utils import * 
from simple_knn._C import distCUDA2 

class GaussianModel:

    def setup_functions(self):
        def build_convariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation) 
            actual_covariance = L @ L.transpose(1,2) 
            symm = strip_symmetric(actual_covariance)
            return symm 
        
        self.scaling_activation = torch.exp 
        self.scaling_inverse_activation = torch.log 

        self.covariance_activation = build_convariance_from_scaling_rotation 

        self.opacity_activation = torch.sigmoid 
        self.inverse_opacity_activation = inverse_sigmoid 

        self.rotation_activation = torch.nn.functional.normalize 

    def __init__(self, sh_degree):
        self.activate_sh_degree = 0 
        self.max_sh_degree = sh_degree 
        self._xyz = torch.empty(0) 
        self._opacity = torch.empty(0) 
        self._scaling = torch.empty(0) 
        self._rotation = torch.empty(0) 
        self._features_dc = torch.empty(0) 
        self._features_rest = torch.empty(0) 

        self.percent_dense = 0 
        self.max_radii2D = torch.empty(0) 
        self.xyz_gradient_accum = torch.empty(0) 
        self.denom = torch.empty(0) 
        self.setup_functions() 

    @property
    def get_xyz(self):
        return self._xyz
    
    @property 
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property 
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property 
    def get_opacity(self):
        return self.opacity_activation(self._opacity) 
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, pcd):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() 
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) 
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1,3) 
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") 
        rots[:, 0] = 1  

        fused_color = RGB2SH(torch.tensor(torch.from_numpy(pcd.colors)).float().cuda()) 
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree+1)**2)).float().cuda()
        features[:,:3,0] = fused_color
        
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        print("Number of points at initialization: ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1,2).contiguous().requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True)) 
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") 
        # TODO: add exposures 

    