import os 
import numpy as np 
import torch 
import torch.nn as nn 
import sys 
from typing import NamedTuple
from read_write_model import * 


def getNerfppNorm(cam_info):
    cam_centers = [] 
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) 
        C2W = np.linalg.inv(W2C) 
        cam_centers.append(C2W[:3, 3:4]) 
    


class BasicPointCloud(NamedTuple):
    points : np.array 
    colors : np.array
    normals : np.array 

class CameraInfo(NamedTuple):
    uid: int 
    R: np.array 
    T: np.array 
    FovY: np.array 
    FovX: np.array 
    depth_params: dict 
    image_path: str 
    image_name: str 
    depth_path: str 
    width: int 
    height: int 
    is_test: bool 

class SceneInfo(NamedTuple):
    point_cloud : BasicPointCloud  
    cameras : list 
    ply_path: str 

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = [] 
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r') 
        # the exact output you're looking for 
        sys.stdout.write("Reading camers {}/{}".format(idx+1, len(cam_extrinsics))) 
        sys.stdout.flush() 

        extr = cam_extrinsics[key] 
        intr = cam_intrinsics[extr.camera_id] 
        height = intr.height 
        width = intr.width 

        uid = intr.id 
        T = np.array(extr.tvec) 
        R = np.transpose(qvec2rotmat(extr.qvec)) 

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] 
            FovY = focal2fov(focal_length_x, height) 
            FovX = focal2fov(focal_length_x, width) 
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0] 
            focal_length_y = intr.params[1] 
            FovY = focal2fov(focal_length_y, height) 
            FovX = focal2fov(focal_length_x, width) 
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name 

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None, 
                              image_path=image_path, image_name=image_name, depth_path=None, 
                              width=width, height=height, is_test=False) 
        cam_infos.append(cam_info) 

    sys.stdout.write('\n') 
    return cam_infos 

def read_colmap_scene(source_path : str, sceneinfo : SceneInfo, images_path:str):
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

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=imageinfos, cam_intrinsics=caminfos,images_folder=images_path
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) 

    bin_path = os.path.join(path, "sparse/0/points3D.bin")

    scene_info = SceneInfo(point_cloud=pcd, cameras=caminfos)

    return scene_info 
    

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
        read_colmap_scene(self.source_path, sceneinfo, images_path)
        

if __name__ == "__main__":
    path = r"D:\code\testVeo3\unzip\south-building"
    Scene(path)