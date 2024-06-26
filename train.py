#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from lpipsPyTorch import lpips
from os import makedirs
from PIL import Image
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, PruneParams
from collections import defaultdict
import yaml
from pathlib import Path
import hashlib
import wandb
import json

from render import render_sets
from metrics import evaluate

from utils.general_utils import inverse_sigmoid
from functools import partial
from utils.prune_utils import _gumbel_sigmoid

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

def training(dataset, opt, pipe, prune, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    prepare_output_and_logger(dataset)
    wandb_enabled = WANDB_FOUND and args.use_wandb
    gaussians = GaussianModel(dataset, prune)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    net_training_time = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, prune.use_mask, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration == opt.densify_until_iter+1:
            gaussians.set_trainable_mask(opt)
            gaussians.opacity_activation = torch.sigmoid
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, (iteration>opt.densify_until_iter) and prune.use_mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            iter_time = iter_start.elapsed_time(iter_end)
            net_training_time += iter_time
            if prune.use_mask:
                log_mask = torch.nn.Threshold(0.5, 0)(gaussians.get_mask)
                sparsity = 1 - torch.count_nonzero(log_mask).cpu().detach().numpy()/torch.numel(log_mask)
            else:
                sparsity = None

            # Log and save
            training_report(wandb_enabled, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), net_training_time, scene, sparsity)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            #Prune
            if iteration in prune.prune_iterations:
                if prune.use_mask:
                    print("prune using mask at ", iteration)
                    prune_mask = gaussians.get_mask < 0.5
                    gaussians.prune_points(prune_mask.squeeze())
                    prune.use_mask = False
                    gaussians.use_mask = False

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if opt.switch_iteration == iteration and opt.use_gumbel_in_finetune:
                print('Switch to Gumbel Sigmoid at iter: ', iteration)
                # gaussians.opacity_activation = torch.sigmoid
                # gaussians.inverse_opacity_activation = inverse_sigmoid
                # gaussians.opacity_activation = partial(_gumbel_sigmoid, temperature = dataset.gumbel_temperature)
                gaussians.opacity_activation = _gumbel_sigmoid


            if opt.use_sigmoid_final and iteration == opt.densify_until_iter:
                gaussians.opacity_activation = torch.sigmoid
                gaussians.inverse_opacity_activation = inverse_sigmoid
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(wandb_enabled, iteration, Ll1, loss, 
                    iter_time, elapsed, scene : Scene, sparsity):

    if wandb_enabled:
        wandb.log({"train_loss_patches/l1_loss": Ll1.item(), 
                   "train_loss_patches/total_loss": loss.item(), 
                   "num_points": scene.gaussians.get_xyz.shape[0],
                   "iter_time": iter_time,
                   "elapsed": elapsed,
                   }, step=iteration)
        if sparsity != None:
            wandb.log({"sparsity": sparsity,
                       }, step = iteration)

if __name__ == "__main__":
    #Set up config file
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, config['model_params'])
    op = OptimizationParams(parser, config['opt_params'])
    pp = PipelineParams(parser, config['pipe_params'])
    pr = PruneParams(parser, config['prune_params'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--retest", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Prepare wandb logger
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)
    pr_args = pr.extract(args)
    id = hashlib.md5(lp_args.wandb_run_name.encode('utf-8')).hexdigest()
    wandb.init(
        project=lp_args.wandb_project,
        name=lp_args.wandb_run_name,
        entity=lp_args.wandb_entity,
        group=lp_args.wandb_group,
        config=args,
        sync_tensorboard=False,
        dir=lp_args.model_path,
        mode=lp_args.wandb_mode,
        id=id,
        resume=True
    )

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not os.path.exists(lp_args.model_path):
                    os.makedirs(lp_args.model_path)
    training(lp_args, op_args, pp_args, pr.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    lp_args.opacity_activation = "sigmoid"

    if not args.skip_test:
        if os.path.exists(os.path.join(args.model_path,"results.json")) and not args.retest:
            print("Testing complete at {}".format(args.model_path))
        else:
            render_sets(lp_args, op_args.iterations, pp_args, pr_args, args.skip_train, args.skip_test)

    evaluate([lp_args.model_path])