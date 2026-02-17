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

import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from time import perf_counter
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Log of testing metrics.
    speeds = []
    qualities = []

    for idx, view in enumerate(tqdm(views, desc="Testing progress")):
        # Speed test.
        if not args.skip_speed:
            deltas = 0
            for i in range(1000):
                # Render.
                pre_render = perf_counter()
                render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
                post_render = perf_counter()

                # Only use last 500.
                if i >= 500:
                    deltas += post_render - pre_render
            speeds.append([idx, 500 / deltas])

        # PSNR test.
        if not args.skip_quality:
            # Render.
            rendering = \
                render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)[
                    "render"]
            gt = view.original_image[0:3, :, :]

            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]

            # Save results.
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            # Compute PSNR.
            mse = torch.mean((gt - rendering) ** 2)
            d_psnr = 10 * torch.log10(1 / mse)
            qualities.append([idx, d_psnr.item()])

    # Write speed data.
    if not args.skip_speed:
        speeds_np = np.array(speeds)
        fps_np = speeds_np[:, 1]
        print(f"FPS:\t{np.mean(fps_np)}\t[{np.min(fps_np)},\t{np.median(fps_np)},\t{np.max(fps_np)}]")
        np.savetxt(os.path.join(base_path, f"{args.branch}--speeds--{args.device}.csv"), speeds_np, delimiter=",",
                   header="Index,FPS", comments="")

    # Write quality data.
    if not args.skip_quality:
        qualities_np = np.array(qualities)
        psnr_np = qualities_np[:, 1]
        print(f"PSNR:\t{np.mean(psnr_np)}\t[{np.min(psnr_np)},\t{np.median(psnr_np)},\t{np.max(psnr_np)}]")
        np.savetxt(os.path.join(base_path, f"{args.branch}--qualities--{args.device}.csv"), qualities_np, delimiter=",",
                   header="Index,PSNR", comments="")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_speed", action="store_true")
    parser.add_argument("--skip_quality", action="store_true")
    parser.add_argument("--branch", default="base", type=str)
    parser.add_argument("--device", default="4090", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                SPARSE_ADAM_AVAILABLE)
