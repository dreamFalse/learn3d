import os 
import numpy as np 
import torch 
import torch.nn as nn 
import sys 
from typing import NamedTuple
from read_write_model import * 
from utils.utils import * 
from camera import Camera 
from gaussian_model import GaussianModel 


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers) 
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True) 
        center = avg_cam_center 
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True) 
        diagonal = np.max(dist) 
        return center.flatten(), diagonal 

    cam_centers = [] 
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) 
        C2W = np.linalg.inv(W2C) 
        cam_centers.append(C2W[:3, 3:4]) 
    
    center, diagonal = get_center_and_diag(cam_centers) 
    radius = diagonal * 1.1 

    translate = -center 

    return {"translate": translate, "radius": radius}


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
    camerainfos : list 
    ply_path: str 
    nerf_normalization : dict 

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

def read_colmap_scene(source_path : str, images_path:str):
    scene_info_path = os.path.join(source_path, "sparse", "0") 

    camera_info_path = os.path.join(scene_info_path, "cameras.bin")
    if os.path.exists(camera_info_path):
        caminfos = read_cameras_binary(camera_info_path)
    else:
        raise NotImplementedError(f"only support binary format, now is {camera_info_path}")

    image_info_path = os.path.join(scene_info_path, "images.bin")
    if os.path.exists(image_info_path):
        imageinfos = read_images_binary(image_info_path)
    else:
        raise NotImplementedError(f"only support binary format, now is {image_info_path}")

    points3d_path = os.path.join(scene_info_path, "points3D.bin")
    if os.path.exists(points3d_path):
        xyzs, rgbs, _ = read_points3D_binary(points3d_path)
        normals = np.zeros_like(xyzs)
        point3d = BasicPointCloud(points=xyzs, colors=rgbs, normals=normals)
    else:
        raise NotImplementedError(f"only support binary format, now is {points3d_path}")

    # print(type(caminfos))
    # print(caminfos)
    # print(type(imageinfos))
    # print(imageinfos)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=imageinfos, cam_intrinsics=caminfos,images_folder=images_path
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) 
    
    nerf_normalization = getNerfppNorm(cam_infos) 

    # TODO: 将point3d存放到本地Ply文件并保存文件路径
    scene_info = SceneInfo(point_cloud = point3d, camerainfos=cam_infos, ply_path=None, nerf_normalization=nerf_normalization)

    return scene_info 


def cameralist_form_caminfos(cam_infos):
    camera_list = [] 
    for cam_info in cam_infos:
        camera_list.append(Camera(cam_info))
    return camera_list 

class Scene:
    gaussians : GaussianModel 
    def __init__(self, source_path, gaussians : GaussianModel):
        self.gaussians = gaussians
        self.source_path = source_path  
        # 从scene_path中读取camera image point cloud信息
        images_path = os.path.join(self.source_path, "images") 
        sceneinfo = read_colmap_scene(self.source_path, images_path)

        # 从camerainfo_list中读取相机内参，外参，C2W matrix, 透视投影矩阵，图像 
        camera_list = cameralist_form_caminfos(sceneinfo.camerainfos)
        self.cameras = camera_list  

        self.camera_extent = sceneinfo.nerf_normalization["radius"]

        self.gaussians.create_from_pcd(sceneinfo.point_cloud) 
        
    def getCameras(self):
        return self.cameras 

if __name__ == "__main__":
    path = r"D:\code\testVeo3\unzip\south-building"
    gaussians = GaussianModel(sh_degree=6)
    Scene(path, gaussians)
