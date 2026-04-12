import os 
import sys 
import torch 
import torch.nn as nn 
from argparse import ArgumentParser
from gaussian_model import GaussianModel 
from scene import Scene 
from tqdm import tqdm 
from random import randint
from gaussian_render import render 

class GroupParams:
    pass 

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none = False):
        group = parser.add_argument_group(name) 
        for key, value in vars(self).items():
            shorthand = False 
            if key.startswith("_"):
                shorthand = True 
                key = key[1:] 
            t = type(value) 
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t) 
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true") 
                else:
                    group.add_argument("--" + key, default=value, type=t) 

    def extract(self, args):
        group = GroupParams() 
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1]) 
        return group 

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 1 
        self._source_path = "" 
        self._model_path = "" 
        self._images = "images" 
        self._depths = "" 
        self._resolution = -1 
        self._white_background = False 
        self.train_test_exp = False 
        self.data_device = "cuda" 
        self.eval = False 
        super().__init__(parser, "Loading Parameters", sentinel) 
    
    def extract(self, args):
        g = super().extract(args) 
        g.source_path = os.path.abspath(g.source_path) 
        return g 
    
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False 
        self.compute_conv3D_python = False 
        self.debug = False 
        self.antialiasing = False 
        super().__init__(parser, "Pipeline Parameters")
    
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000 
        self.position_lr_init = 0.00016 
        self.position_lr_final = 0.0000016 
        self.position_lr_delay_mult = 0.01 
        self.position_lr_max_steps = 30_000 
        self.feature_lr = 0.0025 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005 
        self.rotation_lr = 0.001 
        self.exposure_lr_init = 0.01 
        self.exposure_lr_final = 0.001 
        self.exposure_lr_delay_steps = 0 
        self.exposure_lr_delay_mult = 0.0 
        self.percent_dense = 0.01 
        self.lambda_dssim = 0.2 
        self.densification_interval = 100 
        self.opacity_reset_interval = 3000 
        self.densify_from_iter = 500 
        self.densify_until_iter = 15_000 
        self.densify_grad_threshold = 0.0002 
        self.depth_l1_weight_init = 1.0 
        self.depth_l1_weight_final = 0.01 
        self.random_background = False 
        self.optimizer_type = "default" 
        super().__init__(parser, "Optimization Parameters") 

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    first_iter = 0 
    bg_color = [1,1,1] if dataset.white_background else [0,0,0] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 

    gaussians = GaussianModel(dataset.sh_degree) 
    scene = Scene(dataset, gaussians) 
     
    iter_start = torch.cuda.Event(enable_timing=True) 
    iter_end = torch.cuda.Event(enable_timing=True) 

    viewpoint_stack = scene.getCameras().copy() 
    viewpoint_indices = list(range(len(viewpoint_stack))) 
    ema_loss_for_log = 1.0 
    ema_Ll1depth_for_log = 0.0 

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") 
    first_iter += 1 

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record() 

    # Pick a random Camera 
    rand_idx = randint(0, len(viewpoint_indices)-1) 
    viewpoint_cam = viewpoint_stack.pop(rand_idx) 

    # render 
    bg = torch.rand((3), device="cuda") if opt.random_background else background 

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg) 
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg 

    if viewpoint_cam.alpha_mask is not None:
        alpha_mask = viewpoint_cam.alpha_mask.cuda() 
        image *= alpha_mask 
    
    # loss 
    gt_image = viewpoint_cam.original_image.cuda() 
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainging script parameters") 
    lp = ModelParams(parser) 
    pp = PipelineParams(parser) 
    op = OptimizationParams(parser) 
    parser.add_argument('--ip', type=str, default="127.0.0.1") 
    parser.add_argument('--port', type=int, default=6009) 
    parser.add_argument('--debug_from', type=int, default=-1) 
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[7_000, 30_000]) 
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[7_000, 30_000]) 
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) 
    parser.add_argument('--start_checkpoint', type=str, default=None) 
    args = parser.parse_args(sys.argv[1:]) 

    