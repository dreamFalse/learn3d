import os 
import numpy as np 
import torch 
import torch.nn as nn 
from read_write_model import * 


class BasicPointCloud:
    points : np.array 
    colors : np.array
    normals : np.array 

class SceneInfo:
    point_cloud : BasicPointCloud  
    cameras : list 

def read_colmap_scene(source_path : str, sceneinfo : SceneInfo):
    scene_info_path = os.path.join(source_path, "sparse", "0") 

    camera_info_path = os.path.join(scene_info_path, "cameras.bin")
    if os.path.exists(camera_info_path):
        caminfos = read_cameras_binary(camera_info_path)
    else:
        raise NotImplementedError(f"only support binary format, now is {camera_info_path}")
    print(caminfos)

    image_info_path = os.path.join(scene_info_path, "images.bin")
    if os.path.exists(image_info_path):
        imageinfos = read_images_binary(image_info_path)
    else:
        raise NotImplementedError(f"only support binary format, now is {image_info_path}")

    print(type(imageinfos)) # dict 
    print(imageinfos[1].qvec)
    print(imageinfos[1].tvec)

    points3d_path = os.path.join(scene_info_path, "points3D.bin")
    if os.path.exists(points3d_path):
        points3dinfos = read_points3D_binary(points3d_path)
    else:
        raise NotImplementedError(f"only support binary format, now is {points3d_path}")
    print(type(points3dinfos))
    print(points3dinfos[1])

    

class GaussianModel:
    def __init__(self, sh_degree):
        self.activate_sh_degree = 0 
        self.max_sh_degree = sh_degree 
        self.xyzs = torch.empty() 
        self.opacity = torch.empty() 
        self.scale = torch.empty() 
        self.rotation = torch.empty() 
        self.SHs = torch.empty() 

    def create_from_pcd(self, points3d):
        pass 

class Scene:
    gaussianModel : GaussianModel 
    def __init__(self, source_path):
        self.source_path = source_path  
        # 从scene_path中读取camera image point cloud信息
        images_path = os.path.join(self.source_path, "images") 
        sceneinfo = SceneInfo()
        read_colmap_scene(self.source_path, sceneinfo)
        

if __name__ == "__main__":
    path = r"D:\code\testVeo3\unzip\south-building"
    Scene(path)