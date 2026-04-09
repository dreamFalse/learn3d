import numpy as np 
import os 
import sys 
import torch 
from PIL import Image 
from utils import * 

def PILtoTorch(pil_image):
    image = torch.from_numpy(np.array(pil_image)) / 255.0 
    if len(image.shape) == 3:
        return image.permute(2, 0, 1) 
    else:
        return image.unsqueeze(dim=-1).permute(2, 0, 1) 

class Camera:
    def __init__(self, cam_info, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):

        self.uid = cam_info.uid 
        self.R = cam_info.R 
        self.T = cam_info.T 
        self.FovX = cam_info.FovX 
        self.FovY = cam_info.FovY 
        self.image_name = cam_info.image_name 
        self.image_path = cam_info.image_path 
        
        self.data_device = torch.device("cuda") 

        image = PILtoTorch(Image.open(self.image_path))
        gt_image = image[:3, ...] 
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device) 
        self.image_height = self.original_image.shape[1] 
        self.image_width = self.original_image.shape[2] 

        self.trans = trans 
        self.scale = scale 

        self.z_near = 0.1 
        self.z_far = 100.0 

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda() 
        self.projection_matrix = getProjectionMatrix(znear=self.z_near, zfar = self.z_far, fovX=self.FovX, fovY=self.FovY).transpose(0, 1).cuda() 
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0) 
        self.camera_center = self.world_view_transform.inverse()[3,:3] 

