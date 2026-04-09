import os 
import sys 
import torch 
import numpy as np 
from simple_knn._C import distCUDA2 

class GaussianModel:
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

    def create_from_pcd(self, pcd):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() 
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) 
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1,3) 
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") 
        rots[:, 0] = 1  
        